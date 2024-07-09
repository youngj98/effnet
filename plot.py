import matplotlib.pyplot as plt
import numpy as np
import torch

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
    plt.savefig(f'output_{name}.png')
    plt.show()
