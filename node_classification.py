import pandas as pd
from log import *
from eval import *
from utils import *
from train import *
import gc
from module import GENTI
from graph import NeighborFinder

class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels, shuffle=False):
        temp = list(zip(sources, destinations, timestamps, edge_idxs, labels))
        if shuffle:
            random.shuffle(temp)
        else:
            temp.sort(key=lambda x: x[2], reverse=False)
        sources[:], destinations[:], timestamps[:], edge_idxs[:], labels[:] = zip(*temp)
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = np.concatenate((sources, destinations))
        self.unique_nodes = np.unique(self.unique_nodes)
        self.n_unique_nodes = len(self.unique_nodes)
        self.unique_times = np.unique(timestamps)

def get_data(dataset_name, shuffle=False):
    ### Load data and train val test split
    graph_df = pd.read_csv('./processed/ml_{0}.csv'.format(dataset_name))

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    if {'label'}.issubset(graph_df.columns):
        labels = graph_df.label.values
    else:
        labels = np.ones(sources.shape)
    print('labels: ', labels)
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_mask = timestamps <= val_time #transductive

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask], shuffle=shuffle)

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask], shuffle=shuffle)

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask], shuffle=shuffle)

    pos_mask = (train_data.labels == 1)
    neg_mask = (train_data.labels == 0)

    train_pos_data = Data(train_data.sources[pos_mask], train_data.destinations[pos_mask], train_data.timestamps[pos_mask],
                          train_data.edge_idxs[pos_mask], train_data.labels[pos_mask], shuffle=shuffle)
    train_neg_data = Data(train_data.sources[neg_mask], train_data.destinations[neg_mask], train_data.timestamps[neg_mask],
                          train_data.edge_idxs[neg_mask], train_data.labels[neg_mask], shuffle=shuffle)

    pos_mask = (val_data.labels == 1)
    neg_mask = (val_data.labels == 0)
    
    valid_pos_data = Data(val_data.sources[pos_mask], val_data.destinations[pos_mask], val_data.timestamps[pos_mask],
                          val_data.edge_idxs[pos_mask], val_data.labels[pos_mask], shuffle=shuffle)
    valid_neg_data = Data(val_data.sources[neg_mask], val_data.destinations[neg_mask], val_data.timestamps[neg_mask],
                          val_data.edge_idxs[neg_mask], val_data.labels[neg_mask], shuffle=shuffle)

    pos_mask = (test_data.labels == 1)
    neg_mask = (test_data.labels == 0)
    
    test_pos_data = Data(test_data.sources[pos_mask], test_data.destinations[pos_mask], test_data.timestamps[pos_mask],
                          test_data.edge_idxs[pos_mask], test_data.labels[pos_mask], shuffle=shuffle)
    test_neg_data = Data(test_data.sources[neg_mask], test_data.destinations[neg_mask], test_data.timestamps[neg_mask],
                          test_data.edge_idxs[neg_mask], test_data.labels[neg_mask], shuffle=shuffle)

    print("--------- Get data for node classification: Transductive ---------")
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))

    return full_data, train_pos_data, train_neg_data, valid_pos_data, valid_neg_data, test_pos_data, test_neg_data

if __name__ == '__main__':
    mp.set_start_method(method='forkserver', force=True)
    args, sys_argv = get_args()

    BATCH_SIZE = args.bs
    NUM_WLAKS = args.n_walks
    NUM_EPOCH = args.n_epoch
    ATTN_NUM_HEADS = args.attn_n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    USE_TIME = args.time
    ATTN_AGG_METHOD = args.attn_agg_method
    ATTN_MODE = args.attn_mode
    DATA = args.data
    NUM_STEPS = args.n_steps
    LEARNING_RATE = args.lr
    POS_ENC = args.pos_enc
    POS_DIM = args.pos_dim
    WALK_POOL = args.walk_pool
    WALK_N_HEAD = args.walk_n_head
    WALK_MUTUAL = args.walk_mutual if WALK_POOL == 'attn' else False
    TOLERANCE = args.tolerance
    NGH_CACHE = args.ngh_cache
    VERBOSITY = args.verbosity
    AGG = args.agg
    SEED = args.seed
    set_random_seed(SEED)
    logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)
    device = torch.device('cuda:{}'.format(GPU))

    # Load data and sanity check
    full_data, train_data, train_neg_data, val_data, val_neg_data, test_data, test_neg_data = get_data(DATA)
    src_l = full_data.sources
    dst_l = full_data.destinations
    e_idx_l = full_data.edge_idxs
    ts_l = full_data.timestamps
    train_val_data = (train_data, train_neg_data, val_data, val_neg_data)

    # create two neighbor finders to handle graph extraction.
    # for transductive mode all phases use full_ngh_finder, for inductive node train/val phases use the partial one
    # while test phase still always uses the full one
    n_full = max(max(src_l), max(dst_l))
    full_edges = []
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_edges.append((src, dst, eidx, ts))
        full_edges.append((dst, src, eidx, ts))
        n_full = max(n_full, src, dst)
    full_ngh_finder = NeighborFinder(n_full, full_edges, args.w, bias=args.bias, device=device)

    n_feat = np.load('./processed/ml_{0}_node.npy'.format(DATA))
    e_feat = np.load('./processed/ml_{0}.npy'.format(DATA))

    # model initialization
    genti = GENTI(n_feat, e_feat, agg=AGG,
                num_steps=NUM_STEPS, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
                n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, pos_dim=POS_DIM, pos_enc=POS_ENC,
                num_walks=NUM_WLAKS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL, walk_linear_out=args.walk_linear_out, walk_pool=args.walk_pool,
                verbosity=VERBOSITY, get_checkpoint_path=get_checkpoint_path, device=device)
    genti.to(device)
    optimizer = torch.optim.Adam(genti.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

    # start train and val phases
    logger.info('start train and val phases')
    train_val_node_cls(train_val_data, genti, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, full_ngh_finder, None, logger)

    # final testing
    genti.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch_node_cls('test for {} nodes'.format(args.mode), (test_data, test_neg_data), genti, BATCH_SIZE, eval=False)
    logger.info('Test statistics: {} transductive -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))
    test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1] * 6

    # save model
    logger.info('Saving GENTI model ...')
    torch.save(genti.state_dict(), best_model_path)
    logger.info('GENTI model saved')

    # save one line result
    save_oneline_result('log/', args, [test_acc, test_auc, test_ap, test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc])
