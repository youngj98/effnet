import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score

class_names = ['Normal', 'Snowy', 'Rainy', 'Hazy']

def plot_image_with_predictions(image, predictions, true_label, pred_label, name):
    """
    Plot the image with class probabilities and true label.

    Args:
        image (torch.Tensor): The input image tensor.
        predictions (np.array): The class probabilities.
        true_label (int, optional): The true label of the image. Default: None.
        pred_label (int): The predicted label of the image. Default: None.
        name (str): The name of the output image file.
    """
    # De-normalize the image
    image = image.permute(1, 2, 0).cpu().numpy()
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    
    plt.imshow(image)
    plt.axis('off')
    if true_label is not None:
        text = f"True Label: {class_names[true_label]}\nPrediction Label: {class_names[pred_label]}\n\nClass probabilities:\n"
    else:
        text = f"True Label: None\nPrediction Label: {class_names[pred_label]}\n\nClass probabilities:\n"
    for i, class_name in enumerate(class_names):
        if i < len(class_names) - 1:
            text += f"{class_name}: {predictions[i]:.2f}\n"
        else:
            text += f"{class_name}: {predictions[i]:.2f}"

    plt.text(0.24, 0.6, text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8), transform=plt.gcf().transFigure)
    plt.savefig(f'results/test/output_{name}.png', transparent=True ,bbox_inches='tight')
    plt.show()

def plot_metrics(metrics, save_dir):
    epochs = len(metrics['train_loss'])
    x = range(1, epochs + 1)

    plt.figure()
    plt.plot(x, metrics['train_loss'], label='Train Loss')
    plt.plot(x, metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig(f'{save_dir}/loss_plot.png')
    plt.close()

    plt.figure()
    plt.plot(x, metrics['train_precision'], label='Train Precision')
    plt.plot(x, metrics['val_precision'], label='Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision over Epochs')
    plt.savefig(f'{save_dir}/precision_plot.png')
    plt.close()

    plt.figure()
    plt.plot(x, metrics['train_recall'], label='Train Recall')
    plt.plot(x, metrics['val_recall'], label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('Recall over Epochs')
    plt.savefig(f'{save_dir}/recall_plot.png')
    plt.close()

    plt.figure()
    plt.plot(x, metrics['train_f1'], label='Train F1 Score')
    plt.plot(x, metrics['val_f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score over Epochs')
    plt.savefig(f'{save_dir}/f1_plot.png')
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, save_dir):
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(len(class_names)):
        precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_scores[:, i])
        average_precision[i] = average_precision_score(y_true == i, y_scores[:, i])

    plt.figure()
    for i in range(len(class_names)):
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {class_names[i]} (area = {average_precision[i]:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(f'{save_dir}/precision_recall_curve.png')
    plt.close()
