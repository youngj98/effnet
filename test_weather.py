import os
import torch
import numpy as np
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torch.cuda.amp import autocast
import argparse
from plot import plot_image_with_predictions, class_names


# Transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a custom head for the model to predict climate
class CustomHead(torch.nn.Module):
    def __init__(self, num_ftrs):
        super(CustomHead, self).__init__()
        self.fc_climate = torch.nn.Linear(num_ftrs, 4)  # Assuming 4 classes for climate

    def forward(self, x):
        climate = self.fc_climate(x)
        return climate

# Load the EfficientNet model
def load_model(model_path, device):
    model = EfficientNet.from_name('efficientnet-b5')
    num_ftrs = model._fc.in_features
    model._fc = CustomHead(num_ftrs)
    model._fc = model._fc.to(device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# Load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Main function
def main(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    image = load_image(image_path)
    image = image.to(device)

    with torch.no_grad():
        with autocast():
            outputs_climate = model(image)
        probabilities = torch.softmax(outputs_climate, dim=1).squeeze().cpu().numpy()
        predicted_climate = torch.argmax(outputs_climate, dim=1).item()
        
        file_name = os.path.splitext(os.path.basename(image_path))[0]

        # Plot and print results
        plot_image_with_predictions(image.squeeze(0), probabilities, true_label=None, pred_label=predicted_climate, name=file_name)
        print(f"Predicted Label: {class_names[predicted_climate]}")
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Classification Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model", type=str, default="/home/ailab/AILabDataset/03_Shared_Repository/jonghyun/Project/iitp_bigdata/effnet/best_weather_model_e10_lr0001_s01.pth", help="Path to the model weights")
    args = parser.parse_args()

    main(args.image, args.model)
