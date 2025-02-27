import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Load EfficientNet-B0 model (pretrained on ImageNet)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.eval()

# Remove the classification head to use as a feature extractor
feature_extractor = nn.Sequential(*list(model.children())[:-1])

# Image preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Feature extraction function
def extract_features(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        features = feature_extractor(image)
    return features.squeeze().numpy()

# Save features to a file
def save_features(features, output_file):
    np.save(output_file, features)
    print(f"Features saved to {output_file}")

# Load and visualize feature vector
def visualize_features(feature_file):
    features = np.load(feature_file)
    plt.figure(figsize=(12, 6))
    plt.plot(features, marker='o', linestyle='--', color='b')
    plt.title("Extracted Feature Vector")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.grid()
    plt.show()
    print("Feature vector shape:", features.shape)
    print("First 10 feature values:", features[:10])

# Main function
def main():
    image_path = "scan_image.jpg"  # Replace with actual image path
    output_file = "features.npy"  # File to save features
    
    if not os.path.exists(image_path):
        print("Error: Image file not found!")
        return
    
    features = extract_features(image_path)
    print("Extracted feature vector shape:", features.shape)
    save_features(features, output_file)
    
    # Visualize the extracted features
    visualize_features(output_file)

if __name__ == "__main__":
    main()
