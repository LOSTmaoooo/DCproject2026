import torch
import torch.distributed as dist
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment

"""
Partly Copy-paste from SimGCD. We do some changes to evaluate clustering performance with our code.
https://github.com/CVMI-Lab/SimGCD/blob/main/util/cluster_and_log_utils.py
"""

def all_sum_item(item):
    item = torch.tensor(item).cuda()
    dist.all_reduce(item)
    return item.item()


def split_cluster_acc(y_true, y_pred, mask, idxs, img_paths, is_Test_last):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    Args:
        mask: which instances come from base classes (True) and which ones come from novel classes (False)
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    Return:
        result_dict, the results after matching 
        NMI, in [0, 1]
        ARI, in [-1, 1]
        F1, in [0, 1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1     
    w = np.zeros((D, D), dtype=int)             
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind])
    total_instances = y_pred.size


    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])


    label_new_ls = []
    pred_new_ls = []
 
    result_dict = {}
    new_acc_test = 0
    total_new_instances_test = 0
    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])

        idx = np.where(y_true == i)[0]
        label_new = y_true[idx]
        pred_new = y_pred[idx]
        reverse_map = {v: k for k, v in ind_map.items()}
        reverse_pre_new = [reverse_map[pre] for pre in pred_new]

        label_new_ls.append(label_new)
        pred_new_ls.append(reverse_pre_new)

        if is_Test_last:
            indices = np.where(y_true == i)[0]
            new_idxs = idxs[indices]
            new_img_paths = img_paths[indices]
            predictions = y_pred[indices]
            reverse_ind_map = {v: k for k, v in ind_map.items()}
            matched_pre = [reverse_ind_map[pred] for pred in predictions]

            new_acc_test += matched_pre.count(i)                                   
            total_new_instances_test += len(matched_pre)                           

            result_dict[i] = {'new_idxs': new_idxs, 'img_paths': new_img_paths, 'matched_pre': matched_pre}

    # evaluate metrics
    label_new_ls = np.concatenate(label_new_ls)
    pred_new_ls = np.concatenate(pred_new_ls)

    NMI = metrics.normalized_mutual_info_score(label_new_ls, pred_new_ls)
    ARI = metrics.adjusted_rand_score(label_new_ls, pred_new_ls)
    F1 = metrics.f1_score(label_new_ls, pred_new_ls, average="micro")

    return result_dict, NMI, ARI, F1


def log_accs_from_preds(y_true, y_pred, mask, save_name, idxs, img_paths, T=None,
                        print_output=True, args=None):

    """
    Evaluate clustering performance and log the results

    Args:
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        mask: which instances come from base classes (True) and which ones come from novel classes (False)
        save_name: name to save the results as
        idxs: indices of the images
        img_paths: paths of the images
        T: epoch number
        print_output: whether to print the results
        args: arguments
    """

    is_Test_last = False
    if save_name == 'Test ACC':
        if T + 1 == args.epochs:
            is_Test_last = True

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    idxs = idxs.astype(int)


    result_dict, NMI, ARI, F1 = split_cluster_acc(y_true, y_pred, mask, idxs, img_paths, is_Test_last)
    log_name = f'{save_name}'

    to_return = (result_dict, NMI, ARI, F1)

    if print_output:
        print_str = f'Epoch {T}, {log_name}: NMI {NMI:.4f} | ARI {ARI:.4f} | F1 {F1:.4f}'
        try:
            if dist.get_rank() == 0:
                try:
                    args.logger.info(print_str)
                except:
                    print(print_str)
        except:
            pass

    return to_return