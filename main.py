import pandas as pd
from log import *
from eval import *
from utils import *
from train import *
import gc
from module import GENTI
from graph import NeighborFinder

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
    g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
    if args.data_usage < 1:
        g_df = g_df.iloc[:int(args.data_usage*g_df.shape[0])]
        logger.info('use partial data, ratio: {}'.format(args.data_usage))
    e_feat, n_feat = None, None
    if os.path.exists('./processed/ml_{}.npy'.format(DATA)):
        e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
    if os.path.exists('./processed/ml_{}.npy'.format(DATA)):
        n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))
    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    if 'label' in g_df:
        label_l = g_df.label.values
    else:
        label_l = np.zeros((len(g_df), ), dtype=np.bool_)
    ts_l = g_df.ts.values
    max_idx = max(src_l.max(), dst_l.max())
    max_eidx = e_idx_l.max()
    if e_feat is None:
        e_feat = np.zeros((max_eidx, 1))
    if n_feat is None:
        n_feat = np.random.rand(max_idx + 1, 0)
    assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx or ~math.isclose(1, args.data_usage))  # all nodes except node 0 should appear and be compactly indexed
    assert(n_feat.shape[0] == max_idx + 1 or ~math.isclose(1, args.data_usage))  # the nodes need to map one-to-one to the node feat matrix

    # split and pack the data by generating valid train/val/test mask according to the "mode"
    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
    if args.mode == 't':
        logger.info('Transductive training...')
        valid_train_flag = (ts_l <= val_time)
        valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
        valid_test_flag = ts_l > test_time

    else:
        assert(args.mode == 'i')
        logger.info('Inductive training...')
        # pick some nodes to mask (i.e. reserved for testing) for inductive setting
        total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
        num_total_unique_nodes = len(total_node_set)
        mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
        mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
        mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
        none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
        valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
        valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes
        valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)  # test edges must contain at least one masked node
        valid_test_new_new_flag = (ts_l > test_time) * mask_src_flag * mask_dst_flag
        valid_test_new_old_flag = (valid_test_flag.astype(int) - valid_test_new_new_flag.astype(int)).astype(bool)
        logger.info('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))

    del g_df
    gc.collect()

    # split data according to the mask
    train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], label_l[valid_train_flag]
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], label_l[valid_val_flag]
    test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]
    if args.mode == 'i':
        test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = src_l[valid_test_new_new_flag], dst_l[valid_test_new_new_flag], ts_l[valid_test_new_new_flag], e_idx_l[valid_test_new_new_flag], label_l[valid_test_new_new_flag]
        test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = src_l[valid_test_new_old_flag], dst_l[valid_test_new_old_flag], ts_l[valid_test_new_old_flag], e_idx_l[valid_test_new_old_flag], label_l[valid_test_new_old_flag]
    train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
    val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
    train_val_data = (train_data, val_data)
    del valid_train_flag, valid_val_flag, valid_test_flag
    if args.mode == 'i':
        del valid_test_new_new_flag, valid_test_new_old_flag, mask_src_flag, mask_dst_flag
    gc.collect()
    logger.info('finish split data')

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

    n_partial = 0
    partial_edges = []
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        partial_edges.append((src, dst, eidx, ts))
        partial_edges.append((dst, src, eidx, ts))
        n_partial = max(n_partial, src, dst)
    for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
        partial_edges.append((src, dst, eidx, ts))
        partial_edges.append((dst, src, eidx, ts))
        n_partial = max(n_partial, src, dst)
    partial_ngh_finder = NeighborFinder(n_partial, partial_edges, args.w, bias=args.bias, device=device)
    ngh_finders = partial_ngh_finder, full_ngh_finder

    # create random samplers to generate train/val/test instances
    train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_dst_l, ))
    val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
    test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
    rand_samplers = train_rand_sampler, val_rand_sampler

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
    train_val(train_val_data, genti, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, ngh_finders, rand_samplers, logger)

    # final testing
    genti.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for {} nodes'.format(args.mode), genti, BATCH_SIZE, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l)
    logger.info('Test statistics: {} transductive -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))
    test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1] * 6
    if args.mode == 'i':
        test_new_new_acc, test_new_new_ap, test_new_new_f1, test_new_new_auc = eval_one_epoch('test for {} nodes'.format(args.mode), genti, BATCH_SIZE, test_rand_sampler, test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_label_new_new_l, test_e_idx_new_new_l)
        logger.info('Test statistics: {} inductive -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_new_acc, test_new_new_auc,test_new_new_ap ))

    # save model
    logger.info('Saving GENTI model ...')
    torch.save(genti.state_dict(), best_model_path)
    logger.info('GENTI model saved')

    # save one line result
    save_oneline_result('log/', args, [test_acc, test_auc, test_ap, test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc])
