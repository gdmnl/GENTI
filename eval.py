import math
import torch
import torch.multiprocessing as mp
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from graph import feeder


def eval_one_epoch(hint, tgan, batch_size, sampler, src, dst, ts, label, val_e_idx_l=None, eval=False):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    if eval == False:
        ngh_finder = tgan.ngh_finder
        left, right = mp.Pipe()
        ngh_finder.event = mp.Event()
        ngh_finder.event.set()
        c1 = mp.Process(target=feeder, args=((left, right), ngh_finder.graph, ngh_finder.event))
        c1.start()
        ngh_finder.pipe = left

    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                continue
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            e_l_cut = val_e_idx_l[s_idx:e_idx] if (val_e_idx_l is not None) else None

            size = len(src_l_cut)
            _, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut, test=True)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
    
    if eval == False:
        left.close()
        right.close()
        c1.join()

    return np.mean(val_acc), np.mean(val_ap), None, np.mean(val_auc)