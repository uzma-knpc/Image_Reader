import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

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

# Add this function after extract_features function
def save_features(features, feature_file):
    """Save extracted features to a numpy file"""
    np.save(feature_file, features)
    print(f"Features saved to {feature_file}")

# Compute scalar features
def compute_scalar_features(features):
    mean = np.mean(features)
    std_dev = np.std(features)
    skewness = stats.skew(features)
    kurtosis = stats.kurtosis(features)
    return mean, std_dev, skewness, kurtosis

# Generate clinical report
def generate_report(features, report_file):
    mean, std_dev, skewness, kurtosis = compute_scalar_features(features)
    
    with open(report_file, 'w') as f:
        f.write("Clinical Analysis\n")
        f.write("------------------------------\n")
        f.write("Deep learning features extracted from nuclear imaging data. The extracted feature statistics help analyze uptake patterns.\n\n")
        
        f.write("Procedure\n")
        f.write("------------------------------\n")
        f.write("Images were processed using EfficientNet-B0, and feature vectors were extracted from the penultimate layer. The input images were resized, normalized, and converted into feature representations.\n\n")
        
        f.write("Uptake Value\n")
        f.write("------------------------------\n")
        f.write(f"Mean Uptake: {mean:.4f}\n")
        f.write(f"Standard Deviation: {std_dev:.4f}\n")
        f.write(f"Skewness: {skewness:.4f}\n")
        f.write(f"Kurtosis: {kurtosis:.4f}\n\n")
        
        f.write("Diagnosis\n")
        f.write("------------------------------\n")
        if mean > 0.5 and skewness > 0:
            f.write("Findings suggest hyperfunctioning areas (hot spots), potentially linked to hyperthyroidism, bone metastases, or renal obstruction. Further evaluation is recommended.\n\n")
        elif mean < 0.2 and skewness < 0:
            f.write("Findings suggest hypofunctioning areas (cold spots), which may indicate nodules, renal dysfunction, or bone metastases. Correlation with clinical data is advised.\n\n")
        else:
            f.write("Uptake distribution appears normal. No significant abnormalities detected.\n\n")
        
        f.write("Advice\n")
        f.write("------------------------------\n")
        f.write("Consult a nuclear medicine specialist to correlate findings with clinical history and additional imaging. Machine learning classification may further improve diagnostic accuracy.\n")
    
    print(f"Clinical report saved to {report_file}")

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
    image_path = "/Users/uzmailyas/001003_25.jpg"  # Replace with actual image path
    feature_file = "features.npy"  # File to save features
    report_file = "clinical_report.txt"  # File to save the report
    
    if not os.path.exists(image_path):
        print("Error: Image file not found!")
        return
    
    features = extract_features(image_path)
    print("Extracted feature vector shape:", features.shape)
    
    # Save features using numpy directly
    np.save(feature_file, features)
    print(f"Features saved to {feature_file}")
    
    # Generate a clinical report
    generate_report(features, report_file)
    
    # Visualize the extracted features
    visualize_features(feature_file)

if __name__ == "__main__":
    main()
