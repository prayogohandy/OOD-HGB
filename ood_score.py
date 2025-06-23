import numpy as np
import torch
from metrics import compute_all_metrics
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import tqdm


def print_all_metrics(metrics):
    """
    Print multiple evaluation metrics.

    Parameters:
    -----------
    metrics: list of float
        A list of evaluation metrics in the following order: [fpr, auroc, aupr_in, aupr_out, accuracy].
    """
    
    [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics
    
    print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
            end=' ',
            flush=True)
    print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
        100 * aupr_in, 100 * aupr_out),
            flush=True)
    print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
    print(u'\u2500' * 70, flush=True)

    
def eval_ood(postprocess_results, to_print=False, missclass_as_ood=False):
        """
        Calculates the OOD metrics (fpr, auroc, etc.) based on the postprocessing results.
        
        Parameters:
        -----------
        postprocess_results: list
            A list containing the following elements in order:
            [id_pred, id_conf, ood_pred, ood_conf, id_gt, ood_gt].
        to_print: bool, optional
            Whether to print the evaluation metrics or only return the metrics. Default is True.
        missclass_as_ood: bool, optional
            If True, consider misclassified in-distribution samples as OOD. Default is False.
    
        Returns:
        --------
        dict:
            A dictionary containing various OOD detection evaluation metrics.
        """
    
        [id_pred, id_conf, ood_pred, ood_conf, id_gt, ood_gt] = postprocess_results
        
        if missclass_as_ood:
            id_gt_np = np.array(id_gt)
            id_gt_np[np.array(id_pred) != id_gt_np] = -1
            print((id_gt_np == -1).mean())
            id_gt = id_gt_np.tolist()
            

        pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt, ood_gt])
        
        check_nan = np.isnan(conf)
        num_nans = check_nan.sum()
        if num_nans>0:
            print(num_nans, 'nan ignored.')
            conf = np.delete(conf, np.where(check_nan))
            pred = np.delete(pred, np.where(check_nan))
            label = np.delete(label, np.where(check_nan))
        
        check_inf = np.isinf(conf)
        num_infs = check_inf.sum()
        if num_infs>0:
            print(num_infs, 'inf ignored.')
            conf = np.delete(conf, np.where(check_inf))
            pred = np.delete(pred, np.where(check_inf))
            label = np.delete(label, np.where(check_inf))

        ood_metrics = compute_all_metrics(conf, label, pred)

        if to_print:
            print_all_metrics(ood_metrics)
        else:
            return ood_metrics


def get_ood_score(model, in_test_features, in_test_labels, score_function, out_features=None, missclass_as_ood=False):
    """
    Calculate the novelty scores that an OOD detector (score_function) assigns to ID and OOD and evaluate them via AUROC and FPR.

    Parameters:
    -----------
    model: torch.nn.Module or None
        The neural network model for applying the post-hoc method.
    in_test_features: torch.Tensor
        In-distribution test features.
    in_test_labels: torch.Tensor
        In-distribution test labels.
    score_function: callable
        The scoring function that assigns each sample a novelty score.
    out_features: torch.Tensor or None, optional
        Out-of-distribution (OOD) features. Default is None.
    missclass_as_ood: bool, optional
        If True, consider misclassified in-distribution samples as OOD. Default is False.
    """
    
    if model is not None:
        model.eval() 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # For in-distribution data
    x_in = in_test_features.to(device)
    preds_in, confs_in = score_function(model, x_in)
    gt_in = list(in_test_labels.cpu().detach().numpy())
    
    # For out-of-distribution data
    x_out = out_features.to(device)
    preds_out, confs_out = score_function(model, x_out)
    gt_out = list(np.ones(confs_out.shape[0]) * -1)
    
    return eval_ood([preds_in, confs_in, preds_out, confs_out, gt_in, gt_out], missclass_as_ood=missclass_as_ood)

def get_ood_score_boost(model, in_test_features, in_test_labels, out_features):
    """
    Calculate the novelty scores that an boost method assigns to ID and OOD and evaluate them via AUROC and FPR.

    Parameters:
    -----------
    model: OODModel
        The OODModel class.
    in_test_features: np.ndarray
        In-distribution test features.
    in_test_labels: np.ndarray
        In-distribution test labels.
    out_features: np.ndarray
        Out-of-distribution (OOD) features.
    """

    score = model.predict_proba(in_test_features)
    preds_in = score[:,1:].argmax(1)
    confs_in = score[:,0] # confidence of being ood
    gt_in = in_test_labels # starts at 0

    score = model.predict_proba(out_features)
    preds_out = score[:,1:].argmax(1)
    confs_out = score[:,0]
    gt_out = np.ones(score.shape[0])*-1

    conf = np.concatenate((confs_in, confs_out), axis = 0)
    pred = np.concatenate((preds_in, preds_out), axis = 0)
    label = np.concatenate((gt_in, gt_out), axis = 0)

    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)

    accuracy = acc(pred, label)

    results = [fpr, auroc, aupr_in, aupr_out, accuracy]
    
    return results


# accuracy
def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc

# auc
def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(1 - ood_indicator, 1 - conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(ood_indicator, conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr

# Function to highlight max in one column and min in another
def highlight_ood_metrics(s):
    # Define the tolerance level
    tolerance = 1e-6
    
    if s.name == 'FPR@95':
        # Check if the value is close to the minimum within the tolerance
        is_min = (s <= s.min() + tolerance)
        return ['background-color: lightgreen' if v else '' for v in is_min]
    else:
        # Check if the value is close to the maximum within the tolerance
        is_max = (s >= s.max() - tolerance)
        return ['background-color: lightcoral' if v else '' for v in is_max]