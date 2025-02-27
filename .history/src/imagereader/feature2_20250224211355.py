import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import pytesseract

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

# Extract patient details using OCR
def extract_patient_details(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    details = {"Patient ID": "Unknown", "Patient Name": "Unknown", "Isotope": "Unknown", "Procedure": "Unknown"}
    
    for line in text.split('\n'):
        if "Patient ID" in line:
            details["Patient ID"] = line.split(":")[-1].strip()
        elif "Patient Name" in line:
            details["Patient Name"] = line.split(":")[-1].strip()
        elif "Isotope" in line:
            details["Isotope"] = line.split(":")[-1].strip()
        elif "Procedure" in line:
            details["Procedure"] = line.split(":")[-1].strip()
    
    return details

# Feature extraction function
def extract_features(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        features = feature_extractor(image)
    return features.squeeze().numpy()

# Compute scalar features
def compute_scalar_features(features):
    mean = np.mean(features)
    std_dev = np.std(features)
    skewness = stats.skew(features)
    kurtosis = stats.kurtosis(features)
    return mean, std_dev, skewness, kurtosis

# Compute % uptake from count rate
def compute_uptake(counts_per_sec, baseline_counts_per_sec):
    return (counts_per_sec / baseline_counts_per_sec) * 100 if baseline_counts_per_sec > 0 else 0

# Generate clinical report
def generate_report(features, report_file, patient_details, counts_per_sec, baseline_counts_per_sec):
    mean, std_dev, skewness, kurtosis = compute_scalar_features(features)
    uptake_percentage = compute_uptake(counts_per_sec, baseline_counts_per_sec)
    
    with open(report_file, 'w') as f:
        f.write(f"Patient ID: {patient_details['Patient ID']}\n")
        f.write(f"Patient Name: {patient_details['Patient Name']}\n")
        f.write(f"Isotope Used: {patient_details['Isotope']}\n")
        f.write(f"Procedure: {patient_details['Procedure']}\n\n")
        
        f.write("Clinical Analysis\n")
        f.write("------------------------------\n")
        f.write("Deep learning features extracted from nuclear imaging data. The extracted feature statistics help analyze uptake patterns.\n\n")
        
        f.write("Procedure Details\n")
        f.write("------------------------------\n")
        f.write(f"The patient underwent {patient_details['Procedure']} with {patient_details['Isotope']}. Scans were processed using EfficientNet-B0, and feature vectors were extracted from the penultimate layer.\n\n")
        
        f.write("Uptake Value\n")
        f.write("------------------------------\n")
        f.write(f"Mean Uptake: {mean:.4f}\n")
        f.write(f"Standard Deviation: {std_dev:.4f}\n")
        f.write(f"Skewness: {skewness:.4f}\n")
        f.write(f"Kurtosis: {kurtosis:.4f}\n")
        f.write(f"% Uptake: {uptake_percentage:.2f}%\n\n")
        
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

# Main function
def main():
    image_path = "/Users/uzmailyas/001003_25.jpg"  # Replace with actual image path
    report_file = "clinical_report.txt"  # File to save the report
    
    if not os.path.exists(image_path):
        print("Error: Image file not found!")
        return
    
    # Extract patient details from the image
    patient_details = extract_patient_details(image_path)
    
    # Dummy count rates (Replace with actual values)
    counts_per_sec = 25000
    baseline_counts_per_sec = 50000
    
    features = extract_features(image_path)
    print("Extracted feature vector shape:", features.shape)
    
    # Generate a clinical report
    generate_report(features, report_file, patient_details, counts_per_sec, baseline_counts_per_sec)

if __name__ == "__main__":
    main()
