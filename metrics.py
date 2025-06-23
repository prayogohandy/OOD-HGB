import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, classification_report

def extended_confusion_matrix(y_true, y_pred, y_mask = None, plot = True, class_names = None, prefix=None):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if y_mask is not None:
        cm_mask = confusion_matrix(y_mask, y_pred)
    
    # Calculate precision and recall for each class
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)
    if plot:
        if class_names is None:
            unique_classes = np.unique(y_true)
            class_names = [f'{i}' for i in unique_classes]
    
        # Plot confusion matrix with recall and precision
        total = np.sum(cm)
        labels = [[f"{val:0.0f}\n{val / total:.2%}" for val in row] for row in cm]
        if y_mask is not None:
            labels[0] = [f"{cm_mask[1][i+1]:0.0f}+{cm_mask[0][i+1]:0.0f} \n{val / total:.2%}" for i, val in enumerate(cm[0])]
        
        ax = sns.heatmap(cm, annot=labels, cmap='Reds', fmt='',
                         xticklabels=class_names, yticklabels=class_names, cbar=False)
        ax.set_title('Confusion Matrix', fontweight='bold')
        ax.tick_params(labeltop=True, labelbottom=False, length=0)

        # matrix for the extra column and row
        f_mat = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1))
        f_mat[:-1, -1] = recall  # fill recall column
        f_mat[-1, :-1] = precision  # fill precision row
        f_mat[-1, -1] = accuracy  # accuracy
        
        f_mask = np.ones_like(f_mat)  # puts 1 for masked elements
        f_mask[:, -1] = 0  # last column will be unmasked
        f_mask[-1, :] = 0  # last row will be unmasked
        
        # matrix for coloring the heatmap
        # only last row and column will be used due to masking
        f_color = np.ones_like(f_mat)
        f_color[-1, -1] = 0  # lower right gets different color
        
        # matrix of annotations, only last row and column will be used
        f_annot = [[f"{val:0.2%}" for val in row] for row in f_mat]
        f_annot[-1][-1] = "Accuracy:\n" + f_annot[-1][-1]
        
        sns.heatmap(f_color, mask=f_mask, annot=f_annot, fmt='',
                    xticklabels=class_names + ["Recall"],
                    yticklabels=class_names + ["Precision"],
                    cmap=ListedColormap(['skyblue', 'lightgrey']), cbar=False, ax=ax)

        ax.xaxis.set_label_position('top')
        ax.set_xlabel('Predicted Class', fontweight='bold')
        ax.set_ylabel('Actual Class', fontweight='bold')
        
        if prefix is not None:
            # Save the figure with the specified prefix
            plt.savefig(f"{prefix}_confusion_matrix.jpeg", dpi=600, bbox_inches='tight')
            plt.savefig(f"{prefix}_confusion_matrix.eps", dpi=600, bbox_inches='tight')

        plt.show()
    return (cm, recall, precision, accuracy)


class ClassificationComparison:
    def __init__(self, y_true):
        self.y_true = y_true
        self.method_names = []  # List to store method names
        self.metrics = []  # List to store metrics for each method

    def calculate_metrics(self, y_pred, method_name):
        report = classification_report(self.y_true, y_pred, output_dict=True)
        accuracy = accuracy_score(self.y_true, y_pred)
        
        # Create a dictionary of new metrics
        new_metrics = {
            'Method': method_name,
            'Macro Precision': report['macro avg']['precision'],
            'Macro Recall': report['macro avg']['recall'],
            'Macro F1-Score': report['macro avg']['f1-score'],
            'Accuracy': accuracy
        }
        
        if method_name in self.method_names:
            # Update the metrics for the existing method
            index = self.method_names.index(method_name)
            self.metrics[index] = new_metrics
        else:
            # Append metrics to the list and the method name to the list
            self.metrics.append(new_metrics)
            self.method_names.append(method_name)

    def delete_entry(self, method_name):
        """Delete metrics and predictions for a specified method."""
        if method_name in self.method_names:
            index = self.method_names.index(method_name)
            # Remove from metrics, predictions, and method names
            del self.metrics[index]
            del self.method_names[index]
        else:
            raise ValueError(f"Method '{method_name}' not found.")

    def get_metrics(self, cmap='Blues', highlight_best=False):
        if not self.metrics:
            raise ValueError("No metrics have been calculated yet.")
        
        df = pd.DataFrame(self.metrics).set_index('Method')
        
        if highlight_best:
            return df.style.apply(highlight_best, axis=0)
        
        return df.style.background_gradient(cmap=cmap, axis=0)

def highlight_best(s):
    """Highlight the best values in a Series (column)."""
    tolerance = 1e-6
    is_best = s >= s.max() - tolerance  # Boolean mask for highest values
    return ['background-color: lightgreen' if v else '' for v in is_best]

def compute_all_metrics(conf, label, pred):
    np.set_printoptions(precision=3)
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


# fpr_recall
def fpr_recall(conf, label, tpr):
    gt = np.ones_like(label)
    gt[label == -1] = 0

    fpr_list, tpr_list, threshold_list = metrics.roc_curve(gt, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr)]
    thresh = threshold_list[np.argmax(tpr_list >= tpr)]
    return fpr, thresh


# auc
def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr


# ccr_fpr
def ccr_fpr(conf, fpr, pred, label):
    ind_conf = conf[label != -1]
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    ood_conf = conf[label == -1]

    num_ind = len(ind_conf)
    num_ood = len(ood_conf)

    fp_num = int(np.ceil(fpr * num_ood))
    thresh = np.sort(ood_conf)[-fp_num]
    num_tp = np.sum((ind_conf > thresh) * (ind_pred == ind_label))
    ccr = num_tp / num_ind

    return ccr


def detection(ind_confidences,
              ood_confidences,
              n_iter=100000,
              return_data=False):
    # calculate the minimum detection error
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / n_iter

    best_error = 1.0
    best_delta = None
    all_thresholds = []
    all_errors = []
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        detection_error = (tpr + error2) / 2.0

        if return_data:
            all_thresholds.append(delta)
            all_errors.append(detection_error)

        if detection_error < best_error:
            best_error = np.minimum(best_error, detection_error)
            best_delta = delta

    if return_data:
        return best_error, best_delta, all_errors, all_thresholds
    else:
        return best_error, best_delta
