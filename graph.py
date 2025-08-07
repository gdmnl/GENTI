import gc
import time
import numpy as np
import random
import torch
import math
import matplotlib.pyplot as plt
from sample import *
PRECISION = 5

def feeder(p, graph, event):
    left, right = p
    left.close()
    while True:
        try:
            op, pool, update_list, head, cur_ts = right.recv()
            if op == 'reset':
                graph.reset()
            elif op == 'update':
                L_neighbors, L_eidx, L_ts = pool
                device = L_neighbors.device
                W = L_neighbors.shape[1]

                # Build set of all nodes to update (deduplication)
                update_nodes = set()
                for node_id, cnt in update_list:
                    if node_id != 0 and cnt >= W:
                        update_nodes.add(int(node_id))

                # Add all nodes from edge-triggered update
                updates = []
                k = graph.eidx
                while k < graph.m and graph.edges[k][3] <= cur_ts:
                    n = int(graph.edges[k][0])
                    if n != 0:
                        update_nodes.add(n)
                    updates.append(graph.edges[k])
                    k += 1
                graph.eidx = k
                if len(updates) > 0:
                    graph.insert(updates)

                if not update_nodes:
                    event.set()
                    continue

                rows = []
                samples_n, samples_e, samples_t = [], [], []
                for node_id in sorted(update_nodes):
                    sample_nodes, sample_eidx, sample_ts = graph.sample(node_id, W)
                    if len(sample_nodes) > 0:
                        rows.append(node_id)
                        samples_n.append(sample_nodes)
                        samples_e.append(sample_eidx)
                        samples_t.append(sample_ts)

                if rows:
                    rows_tensor = torch.tensor(rows, dtype=torch.long, device=device)
                    arr_n = torch.from_numpy(np.stack(samples_n)).to(device, non_blocking=True)
                    arr_e = torch.from_numpy(np.stack(samples_e)).to(device, non_blocking=True)
                    arr_t = torch.from_numpy(np.stack(samples_t)).to(device, non_blocking=True)
                    L_neighbors[rows_tensor, :] = arr_n
                    L_eidx[rows_tensor, :]      = arr_e
                    L_ts[rows_tensor, :]        = arr_t

                event.set()
            else:
                raise NotImplementedError('invalid operator')
        except EOFError:
            right.close()
            break

class Bucket(object):
    def __init__(self, idx, nxt):
        self.sum = 0
        self.idx = idx
        self.nxt = nxt
        self.table = {}
        self.items = []
    
    def insert(self, item, weight):
        self.items.append((item, weight))
        self.table[item] = len(self.items) - 1

        self.sum += math.exp(weight)
    
    def delete(self, item, weight):
        pos = self.table[item]
        self.table[self.items[-1]] = pos
        self.table.pop(self.items[pos])

        self.items[pos], self.items[-1] = self.items[-1], self.items[pos]
        self.items = self.items[: -1]
        
        self.sum -= math.exp(weight)
    
    def sample(self, num):
        ret = []
        for _ in range(num):
            length = len(self.items)
            while True:
                r = random.random()
                item, weight = self.items[random.randint(0, length - 1)]
                if r < math.exp(weight - self.idx - 1):
                    ret.append(item)
                    break
        return ret


class Graph:
    def __init__(self, alpha, edges, W, N):
        self.alpha = alpha
        self.edges = edges
        self.m = len(edges)
        self.n = N
        self.W = W
        self.eidx = 0
        self.top = [-1 for _ in range(N)]
        self.sum = [0 for _ in range(N)]

    def insert(self, reqs):
        for req in reqs:
            u, v, e_idx, t = req
            weight = t * self.alpha
            idx = math.floor(weight)
            bkt = self.top[u]
            if bkt == -1 or bkt.idx != idx:
                bkt = Bucket(idx, self.top[u])
                self.top[u] = bkt

            bkt.insert((v, e_idx, t), weight)
            self.sum[u] += math.exp(weight)

    def sample(self, u, num):
        queries = np.random.rand(num) * self.sum[u]
        queries = sorted(queries, reverse=True)

        ret = []
        l, r = 0, 0
        sum = self.sum[u]
        cur = self.top[u]
        while cur != -1:
            sum -= cur.sum
            while r < num and queries[r] > sum:
                r += 1
            ret.extend(cur.sample(r - l))
            l = r
            cur = cur.nxt

        b_nodes = torch.tensor([r[0] for r in ret]).long()
        b_idx = torch.tensor([r[1] for r in ret]).long()
        b_ts = torch.tensor([r[2] for r in ret]).float()
        return b_nodes, b_idx, b_ts

    def reset(self):
        self.eidx = 0
        for u in range(self.n):
            cur = self.top[u]
            while cur != -1:
                nxt = cur.nxt
                del cur
                cur = nxt
            self.top[u] = -1
            self.sum[u] = 0
        gc.collect()


class NeighborFinder:
    def __init__(self, n, edges, W=16, bias=0, device='cpu'):
        self.bias = bias  # the "alpha" hyperparameter
        self.edges = edges
        self.m = len(edges)
        self.n = n + 1
        self.graph = Graph(bias, edges, W, self.n)

        self.W = W
        self.L_neighbors = torch.from_numpy(np.zeros((self.n, self.W))).long().to(device)
        self.L_eidx = torch.from_numpy(np.zeros((self.n, self.W))).long().to(device)
        self.L_ts = torch.from_numpy(np.zeros((self.n, self.W))).float().to(device)
        self.head = torch.from_numpy(np.zeros((self.n, ))).long().to(device)
        self.eidx = 0

        self.pipe = None
        self.event = None

        self.total_consumed = torch.zeros(self.n, dtype=torch.long, device=device)

    def update_async(self, cur_ts):
        """
        A updated strategy: instead of updating whenever there are pending neighbors,
        we now batch and only submit a full-row update when a node accumulates W updates,
        enabling more efficient GPU processing.
        """
        enough_mask = self.total_consumed >= self.W
        nodes_to_update = enough_mask.nonzero(as_tuple=False).squeeze(-1)
        if nodes_to_update.numel() == 0:
            return 

        update_counts = torch.full_like(nodes_to_update, self.W)
        update_list = list(zip(nodes_to_update.tolist(), update_counts.tolist()))

        self.pipe.send(('update', (self.L_neighbors, self.L_eidx, self.L_ts), update_list, self.head.cpu().numpy(), cur_ts))
        self.event.clear()

        self.total_consumed[nodes_to_update] = 0

    def gather_l_hop_walks(self, L, src_nodes, cur_time, n_walk=100, e_idx=None):
        '''
        Gather L-hop walks based on the sampled neighbor pool to form a subgraph.
        Update: uses torch.unique_consecutive for efficient local index calculation.
        '''
        Q = len(src_nodes)
        if Q == 0:
            return None, None, None

        device = src_nodes.device
        cur_nodes = src_nodes.reshape(-1, 1).repeat(1, n_walk).reshape(-1)
        cur_time = cur_time.reshape(-1, 1).repeat(1, n_walk).reshape(-1)
        n_idx_batch, e_idx_batch, ts_batch = [cur_nodes], [], [cur_time]
        T = len(cur_nodes)
        if e_idx is None:
            e_idx_batch.append(torch.ones_like(cur_nodes).long().to(device) * (-1))
        else:
            e_idx = e_idx.reshape(-1, 1).repeat(1, n_walk).reshape(-1)
            e_idx_batch.append(e_idx)

        for _ in range(L):
            # Efficient per-group range calculation (O(N)), assumes cur_nodes is grouped
            unique_nodes, counts = torch.unique_consecutive(cur_nodes, return_counts=True)
            group_starts = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, 0)[:-1]])
            total = counts.sum().item()
            pos_in_total = torch.arange(total, device=device)
            local_index = pos_in_total - torch.repeat_interleave(group_starts, counts)

            pool_index = (self.head[cur_nodes] + local_index) % self.W

            cur_ts   = self.L_ts[cur_nodes, pool_index]
            cur_eidx = self.L_eidx[cur_nodes, pool_index]
            next_nodes = self.L_neighbors[cur_nodes, pool_index]

            n_idx_batch.append(next_nodes)
            e_idx_batch.append(cur_eidx)
            ts_batch.append(cur_ts)

            self.total_consumed.scatter_add_(0, unique_nodes, counts)
            self.head[unique_nodes] = (self.head[unique_nodes] + counts) % self.W

            cur_nodes = next_nodes

        L += 1
        n_idx_batch = torch.stack(n_idx_batch, dim=-1).reshape(Q, n_walk, L)
        e_idx_batch = torch.stack(e_idx_batch, dim=-1).reshape(Q, n_walk, L)
        ts_batch = torch.stack(ts_batch, dim=-1).reshape(Q, n_walk, L)

        return n_idx_batch, e_idx_batch, ts_batch
