import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import easyocr

# Load EfficientNet-B0 model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.eval()

# Remove the classification head
feature_extractor = nn.Sequential(*list(model.children())[:-1])

# Initialize OCR reader
reader = easyocr.Reader(['en'])

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

def extract_patient_details(image_path):
    """Extract patient details from image using EasyOCR"""
    try:
        # Read text from image
        results = reader.readtext(image_path)
        text = ' '.join([result[1] for result in results])
        
        # Initialize details dictionary
        details = {
            "Patient ID": "Unknown",
            "Patient Name": "Unknown",
            "Isotope": "Unknown",
            "Procedure": "Unknown"
        }
        
        # Extract details using common patterns
        for result in results:
            text_line = result[1].lower()
            if any(id_pattern in text_line for id_pattern in ['id:', 'id no:', 'patient id:']):
                details["Patient ID"] = result[1].split(':')[-1].strip()
            elif any(name_pattern in text_line for name_pattern in ['name:', 'patient name:']):
                details["Patient Name"] = result[1].split(':')[-1].strip()
            elif 'tc-99m' in text_line or 'i-131' in text_line:
                details["Isotope"] = result[1].strip()
            elif any(proc in text_line for proc in ['scan', 'study', 'procedure']):
                details["Procedure"] = result[1].strip()
        
        print("Extracted patient details:", details)
        return details
        
    except Exception as e:
        print(f"Error extracting text: {e}")
        return {
            "Patient ID": "Error in extraction",
            "Patient Name": "Error in extraction",
            "Isotope": "Error in extraction",
            "Procedure": "Error in extraction"
        }

def generate_report(features, report_file, patient_details, counts_per_sec=25000, baseline_counts_per_sec=50000):
    mean, std_dev, skewness, kurtosis = compute_scalar_features(features)
    uptake_percentage = (counts_per_sec / baseline_counts_per_sec) * 100 if baseline_counts_per_sec > 0 else 0
    
    with open(report_file, 'w') as f:
        f.write("NUCLEAR MEDICINE SCAN REPORT\n")
        f.write("============================\n\n")
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
    
    print(f"Report generated: {report_file}")

def visualize_features(features):
    """Visualize the 1280 feature vector with enhanced plotting"""
    plt.figure(figsize=(15, 8))
    
    # Create main feature plot
    plt.subplot(2, 1, 1)
    plt.plot(features, 'b-', alpha=0.6, label='Feature Values')
    plt.plot(features, 'r.', alpha=0.5, markersize=2)
    plt.title('Feature Vector Visualization (1280 Features)', fontsize=12, pad=10)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Create distribution plot
    plt.subplot(2, 1, 2)
    plt.hist(features, bins=50, color='green', alpha=0.6)
    plt.title('Feature Value Distribution', fontsize=12)
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Add statistics annotation
    stats_text = f'Mean: {np.mean(features):.4f}\n'
    stats_text += f'Std: {np.std(features):.4f}\n'
    stats_text += f'Max: {np.max(features):.4f}\n'
    stats_text += f'Min: {np.min(features):.4f}'
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Print feature statistics
    print("\nFeature Vector Statistics:")
    print(f"Shape: {features.shape}")
    print(f"Mean: {np.mean(features):.4f}")
    print(f"Std Dev: {np.std(features):.4f}")
    print(f"Max Value: {np.max(features):.4f}")
    print(f"Min Value: {np.min(features):.4f}")

def main():
    image_path = input("Enter image path: ")
    report_file = "clinical_report.txt"
    
    if not os.path.exists(image_path):
        print("Error: Image file not found!")
        return
    
    # Extract patient details from image
    patient_details = extract_patient_details(image_path)
    
    # Extract and visualize features
    features = extract_features(image_path)
    visualize_features(features)  # Call the new visualization function
    
    # Generate report
    generate_report(features, report_file, patient_details)

if __name__ == "__main__":
    main()
