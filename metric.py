import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def precision_recall_f1score(preds, labels, average='macro'):
    """
    Calculate precision, recall, and f1 score.

    Args:
        preds (np.array): Predictions from the model.
        labels (np.array): True labels.
        average (str): The type of averaging performed on the data.
                       Options are ['macro', 'micro', 'weighted', 'samples'].

    Returns:
        precision (float): Calculated precision.
        recall (float): Calculated recall.
        f1_score (float): Calculated f1 score.
    """
    # Ensure inputs are numpy arrays
    preds = np.array(preds)
    labels = np.array(labels)

    # Initialize metrics
    precision, recall, f1_score = 0.0, 0.0, 0.0

    if average == 'macro':
        # Calculate per-class precision, recall and f1 score
        for cls in np.unique(labels):
            tp = ((preds == cls) & (labels == cls)).sum()
            fp = ((preds == cls) & (labels != cls)).sum()
            fn = ((preds != cls) & (labels == cls)).sum()

            cls_precision = tp / (tp + fp) if tp + fp > 0 else 0
            cls_recall = tp / (tp + fn) if tp + fn > 0 else 0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if cls_precision + cls_recall > 0 else 0

            precision += cls_precision
            recall += cls_recall
            f1_score += cls_f1

        # Average the metrics
        precision /= len(np.unique(labels))
        recall /= len(np.unique(labels))
        f1_score /= len(np.unique(labels))

    elif average == 'micro':
        # Calculate global tp, fp, fn
        tp = (preds == labels).sum()
        fp = (preds != labels).sum()
        fn = (preds != labels).sum()

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    elif average == 'weighted':
        class_counts = np.bincount(labels)
        total = len(labels)
        for cls in np.unique(labels):
            tp = ((preds == cls) & (labels == cls)).sum()
            fp = ((preds == cls) & (labels != cls)).sum()
            fn = ((preds != cls) & (labels == cls)).sum()

            cls_precision = tp / (tp + fp) if tp + fp > 0 else 0
            cls_recall = tp / (tp + fn) if tp + fn > 0 else 0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if cls_precision + cls_recall > 0 else 0

            precision += cls_precision * (class_counts[cls] / total)
            recall += cls_recall * (class_counts[cls] / total)
            f1_score += cls_f1 * (class_counts[cls] / total)

    return precision, recall, f1_score

def plot_confusion_matrix(true_labels, pred_labels, classes, train_setting, name):
    """
    Plot confusion matrix using true and predicted labels.

    Args:
        true_labels (np.array): True labels.
        pred_labels (np.array): Predicted labels.
        classes (list): List of class names.
    """
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    # Save the plot to a file called confusion_matrix_name.png
    plt.savefig(f'results/train/{train_setting}/confusion_matrix_{name}.png')
    plt.show()
