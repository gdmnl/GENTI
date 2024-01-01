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
            op, pool, cur_ts = right.recv()        
            if op == 'reset':
                graph.reset()
            elif op == 'update':
                updates = []
                reqs = []
                k = graph.eidx
                while k < graph.m and graph.edges[k][3] <= cur_ts:
                    reqs.append(graph.edges[k][0])
                    updates.append(graph.edges[k])
                    k += 1
                graph.eidx = k
                graph.insert(updates)

                torch.cuda.synchronize()
                p = time.time()
                samples_nodes = []
                samples_eidx = []
                samples_ts = []
                reqs = np.unique(reqs)
                for req in reqs:
                    sample = graph.sample(req, graph.W)
                    samples_nodes.append(sample[0])
                    samples_eidx.append(sample[1])
                    samples_ts.append(sample[2])
                device = pool[0].device
                pool[0][reqs, :] = torch.stack(samples_nodes, dim=0).to(device, non_blocking=True)
                pool[1][reqs, :] = torch.stack(samples_eidx, dim=0).to(device, non_blocking=True)
                pool[2][reqs, :] = torch.stack(samples_ts, dim=0).to(device, non_blocking=True)

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
        b_ts =torch.tensor([r[2] for r in ret]).float()
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
        self.head = torch.from_numpy(np.zeros((self.n, self.W))).long().to(device)
        self.eidx = 0

        self.pipe = None
        self.event = None

    def update_async(self, cur_ts):
        self.pipe.send(('update', (self.L_neighbors, self.L_eidx, self.L_ts), cur_ts))
        self.event.clear()

    def gather_l_hop_walks(self, L, src_nodes, cur_time, n_walk=100, e_idx=None):
        '''
        gather L_hop walks based on sampled neighbor pool to form a subgraph.
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
            idx_pos = torch.randint(0, self.W, (T, )).long().to(device)
            cur_ts = self.L_ts[cur_nodes, idx_pos]
            cur_eidx = self.L_eidx[cur_nodes, idx_pos]
            cur_nodes = self.L_neighbors[cur_nodes, idx_pos]
            n_idx_batch.append(cur_nodes)
            e_idx_batch.append(cur_eidx)
            ts_batch.append(cur_ts)

        L += 1
        n_idx_batch = torch.stack(n_idx_batch, dim=-1).reshape(Q, n_walk, L)
        e_idx_batch = torch.stack(e_idx_batch, dim=-1).reshape(Q, n_walk, L)
        ts_batch = torch.stack(ts_batch, dim=-1).reshape(Q, n_walk, L)

        return n_idx_batch, e_idx_batch, ts_batch
