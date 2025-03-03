import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import google.generativeai as genai
import os
#from facenet_pytorch import InceptionResnetV1
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import shutil
import requests
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import datetime
from PIL import Image
from transformers import pipeline
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from scipy import stats  # Add this import

from dotenv import load_dotenv

load_dotenv()
#image_path = './images/thyaemc.jpeg'
#image_path=os.path(Image_path)
class practice:
    def __init__(self):
        # Create necessary directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Initialize model
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.eval()
        
        # Save model to models directory
        model_path = os.path.join("models", "fine_tuned_inception_resnet_v1.pth")
        torch.save(self.model.state_dict(), model_path)
        
        self.image_path = None

        # Add prompt1 for patient details
        self.prompt1 = """This image contains a human organ image along with notes and graph.
    Given the Medical image, extract the following patient details:

    PATIENT DETAILS:
    - Name
    - Patient ID/PRN-NO
    - Age/Gender
    - Isotope Used
    - Scan Date
    - Referring Physician
    - Clinical History
    - Procedure Type

    Return the extracted information in a structured format.
    """

        # Main analysis prompt will be updated in diagnose_image with feature values
        self.prompt = ""  # This will be set with actual values in diagnose_image

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_gen = genai.GenerativeModel("gemini-1.5-flash")
        # Remove ONNX initialization
        # self.session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

    def get_file_from_user(self):
        """
        Get file path from user input or URL and load the file
        Returns:
            tuple: (file_paths, titles)
        """
        try:
            # Get file path or URL from user
            source = input("Input the path to your file or URL: ")
            
            # Create uploads directory if it doesn't exist
            os.makedirs("uploads", exist_ok=True)
            
            # Get filename
            if source.startswith(("http://", "https://")):
                # Handle URL
                filename = source.split('/')[-1]  # Get filename from URL
                destination = os.path.join("uploads", filename)
                
                # Download the image
                response = requests.get(source, stream=True)
                response.raise_for_status()
                
                # Save the image
                with open(destination, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                print(f"Image downloaded and saved to: {destination}")
                
            else:
                # Handle local file
                if not os.path.exists(source):
                    raise FileNotFoundError(f"File not found: {source}")
                    
                filename = os.path.basename(source)
                destination = os.path.join("uploads", filename)
                shutil.copy2(source, destination)
                print(f"File copied to: {destination}")
            
            print(f"File path is: {destination}")
            self.image_path = destination
            return [self.image_path], [filename]
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    #process file into normalized tensor
    def preprocess_image(self, image_path=None):
        if image_path is None:
            image_path = self.image_path
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transformed_tensor = transform(image).unsqueeze(0)
        # Make sure output is compatible with ONNX model input
        return transformed_tensor  # Shape should match model input

    # Function to Extract Features
    def extract_features(self, image_tensor=None):
        if image_tensor is None:
            image_tensor = self.preprocess_image()
        
        with torch.no_grad():
            features = self.model(image_tensor)
            features_np = features.numpy().flatten()
            
            # Add visualization
            self.visualize_features(features_np)
            
            return features_np

    # Function to create image embeddings
    def create_image_embedding(self, image_path=None):
        if image_path is None:
            image_path = self.image_path
        try:
            input_tensor = self.preprocess_image(image_path)
            print(f"image path{image_path}")
            with torch.no_grad():
                embeddings = self.model(input_tensor)
                return embeddings.squeeze().numpy()
        except Exception as e:
            print("Error:", e)
            return None

    # Save the fine-tuned model
    #torch.save(model.state_dict(), "fine_tuned_inception_resnet_v1.pth")
        # Example usage:
        # Image_path = get_file_from_user()
        # if Image_path:
        #     process_image(Image_path)
        
    # Function to load an image and convert it to grayscale
    def load_image(self, file_path):
        img = Image.open(file_path).convert("L")
        img_array = np.array(img)
        return img_array

    # Function to normalize the image intensity
    def normalize_image(self, img):
        img_min, img_max = img.min(), img.max()
        normalized_img = (img - img_min) / (img_max - img_min)
        return normalized_img
    # Diagnostic function with criteria
    def diagnose_image(self, img, features):
        """Diagnose and prepare feature analysis"""
        # Calculate all metrics
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        uptake_percentage = np.sum(img > 0.5) / img.size
        
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        feature_kurtosis = stats.kurtosis(features)
        feature_skewness = stats.skew(features)
        
        # Store metrics
        self.diagnosis_metrics = {
            "Mean Intensity": mean_intensity,
            "Standard Deviation": std_intensity,
            "Uptake Percentage": uptake_percentage,
            "Feature Mean": feature_mean,
            "Feature Std": feature_std,
            "Feature Kurtosis": feature_kurtosis,
            "Feature Skewness": feature_skewness
        }
        
        # Update prompt with actual values
        self.prompt = f"""This image contains a human organ image along with notes and graph.
    Analyze the image based on these measured feature values and their clinical implications:

    MEASURED FEATURE VALUES:
    1. Feature Mean: {feature_mean:.4f}
       - Normal Range: 0.3 to 0.6
       - Current Status: {"High (Hyperactive)" if feature_mean > 0.6 else "Low (Hypoactive)" if feature_mean < 0.3 else "Normal"}
    
    2. Feature Kurtosis: {feature_kurtosis:.4f}
       - Normal Range: -2.0 to 2.0
       - Current Status: {"High (Focal)" if feature_kurtosis > 2.0 else "Low (Diffuse)" if feature_kurtosis < -1.0 else "Normal"}
    
    3. Feature Skewness: {feature_skewness:.4f}
       - Normal Range: -0.5 to 0.5
       - Current Status: {"Positive (Hot Spots)" if feature_skewness > 0.5 else "Negative (Cold Spots)" if feature_skewness < -0.5 else "Normal"}
    
    4. Feature Standard Deviation: {feature_std:.4f}
       - Normal Range: 0.15 to 0.25
       - Current Status: {"High Variability" if feature_std > 0.25 else "Low Variability" if feature_std < 0.15 else "Normal"}

    Based on these specific values, provide:

    DESCRIPTION: {{
        "Organ_Type": "Describe the organ visible in image",
        "Pattern_Analysis": "Analyze uptake pattern based on measured feature mean {feature_mean:.4f} and kurtosis {feature_kurtosis:.4f}",
        "Distribution_Type": "Determine if uptake is focal (high kurtosis) or diffuse (low kurtosis)",
        "Key_Features": "List features based on measured values"
    }}

    PREDICTION: {{
        "Primary_Condition": "Determine condition based on feature mean {feature_mean:.4f} and skewness {feature_skewness:.4f}",
        "Confidence_Level": "Assess based on feature clarity and standard deviation {feature_std:.4f}",
        "Supporting_Evidence": "Use measured feature values as evidence"
    }}

    ABNORMALITIES: {{
        "Hot_Spots": "Analyze based on positive skewness {feature_skewness:.4f if feature_skewness > 0 else 'N/A'}",
        "Cold_Spots": "Analyze based on negative skewness {abs(feature_skewness):.4f if feature_skewness < 0 else 'N/A'}",
        "Pattern_Irregularities": "Identify based on kurtosis {feature_kurtosis:.4f} and std {feature_std:.4f}"
    }}

    QUANTITATIVE MEASUREMENTS: {{
        "Feature_Statistics": {{
            "Mean_Value": "{feature_mean:.4f} - {'High' if feature_mean > 0.6 else 'Low' if feature_mean < 0.3 else 'Normal'}",
            "Kurtosis": "{feature_kurtosis:.4f} - {'Focal' if feature_kurtosis > 2.0 else 'Diffuse' if feature_kurtosis < -1.0 else 'Normal'}",
            "Skewness": "{feature_skewness:.4f} - {'Hot Spots' if feature_skewness > 0.5 else 'Cold Spots' if feature_skewness < -0.5 else 'Normal'}",
            "Standard_Deviation": "{feature_std:.4f} - {'High Variability' if feature_std > 0.25 else 'Low Variability' if feature_std < 0.15 else 'Normal'}"
        }},
        "Clinical_Significance": "Interpret these specific values in clinical context",
        "Comparison_To_Normal": "Compare measured values to normal ranges provided"
    }}

    Return a detailed analysis focusing on these specific measured values and their clinical implications.
    """
        
        # Return diagnosis based on feature values
        if (feature_mean > 0.6 and feature_kurtosis > 2.0):
            return "HIGH UPTAKE DETECTED\n- Feature analysis indicates significant abnormal patterns..."
        # ... rest of the diagnosis logic ...

    # Function to calculate image metrics
    def calculate_metrics(self, img):
        flat_img = img.flatten()  # Flatten array for stats calculations
        return {
            "Mean Intensity": np.mean(img),
            "Standard Deviation": np.std(img),
            "Minimum Intensity": np.min(img),
            "Maximum Intensity": np.max(img),
            "Skewness": stats.skew(flat_img),  # Use scipy.stats.skew
            "Kurtosis": stats.kurtosis(flat_img),  # Use scipy.stats.kurtosis
            "Uptake Percentage": np.sum(img > 0.5) / img.size
        }
        image = Image.open(Image_path)  # Load the image
    #image

    def generate_content(self):
        """Generate content using both prompts"""
        try:
            image = Image.open(self.image_path)
            
            # Generate patient details first
            self.response_gen1 = self.model_gen.generate_content([self.prompt1, image])
            
            # Then generate analysis using feature-based prompt
            self.response_gen = self.model_gen.generate_content([self.prompt, image])
            
            return self.response_gen, self.response_gen1
        except Exception as e:
            print(f"Error generating content: {e}")
            return None

    # Function to generate a medical report
    def generate_report(self, scan_id, scan_name, diagnosis, features, doctor_name):
        """Generate medical report using both diagnosis metrics and feature statistics"""
        report = f"""
# 🏥 MEDICAL IMAGING ANALYSIS REPORT
## Atomic Energy Cancer Hospital, PAKISTAN
### AI-Assisted Image Analysis Report

---

📋 **Report Details**
- **Report ID:** {scan_id}
- **Scan Name:** {scan_name}
- **Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

🔍 **Diagnosis**
{diagnosis}

👤 **Patient Details**
{self.response_gen1.text}

📊 **Clinical Measurements**
Image Metrics:
- Mean Intensity: {self.diagnosis_metrics["Mean Intensity"]:.4f}
- Standard Deviation: {self.diagnosis_metrics["Standard Deviation"]:.4f}
- uptake Percentage: {self.diagnosis_metrics["Uptake Percentage"]:.4f}

Feature Statistics:
- Feature Mean: {self.diagnosis_metrics["Feature Mean"]:.4f}
- Feature Std: {self.diagnosis_metrics["Feature Std"]:.4f}
- Feature Kurtosis: {self.diagnosis_metrics["Feature Kurtosis"]:.4f}
- Feature Skewness: {self.diagnosis_metrics["Feature Skewness"]:.4f}

-------------------------------------------------
***Standard Values***
- Normal: -2 < Feature Kurtosis < 2, 0.3 < Mean Intensity < 0.5
- Hot spots: Feature Kurtosis > 2.0, Mean Intensity > 0.5
- Cold spots: Feature Kurtosis < -1.0, Mean Intensity < 0.3

📝 **Analysis and Findings**
{self.response_gen.text}

✅ **Authentication**
- **Reporting Doctor:** {doctor_name}
- **Report Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

                                                    Head of Department
                                                    Consultant Nuclear Physician
                                                    Atomic Energy Cancer Hospital
"""
        return report

    def process_and_generate_reports(self, file_paths, titles, doctor_name):
        reports = []  # Initialize an empty list to store the reports
        rows, cols = 2, 3
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        axes = axes.flatten()

        for i, (file_path, title) in enumerate(zip(file_paths, titles)):
            if i >= len(axes):
                break

            img = self.load_image(file_path)
            imgname=titles[i]
            normalized_img = self.normalize_image(img)
            features = self.extract_features()
            diagnosis = self.diagnose_image(normalized_img, features)
            metrics = self.calculate_metrics(normalized_img)
            scan_id = i + 1

            # Append report
            report = self.generate_report(scan_id, title, diagnosis, features, doctor_name)
            reports.append(report)

            # Display image with diagnosis
            ax = axes[i]
            im = ax.imshow(normalized_img, cmap="gray")
            ax.set_title(f"{title}\n{diagnosis}", fontsize=10)
            ax.axis("off")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # Add colorbar
        plt.tight_layout()
        # Remove plt.show() from here
        # plt.show()  # Comment out or remove this line

        return reports  # Return the generated reports list
    
    # Add this after calculate_metrics method
    def visualize_features(self, features):
        """Visualize feature vector with enhanced plotting"""
        plt.figure(figsize=(15, 8))
        
        # Create main feature plot with grid
        plt.subplot(2, 1, 1)
        plt.plot(features, 'b-', alpha=0.6, label='Feature Values')
        plt.plot(features, 'r.', alpha=0.5, markersize=2)
        plt.title('Nuclear Image Feature Analysis', fontsize=14, pad=10)
        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Feature Value', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=10)

        # Create distribution histogram
        plt.subplot(2, 1, 2)
        plt.hist(features, bins=50, color='green', alpha=0.6, density=True)
        plt.title('Feature Value Distribution', fontsize=12)
        plt.xlabel('Feature Value', fontsize=10)
        plt.ylabel('Density', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')

        # Add statistical annotations
        stats_text = (
            f'Statistical Analysis:\n'
            f'Mean: {np.mean(features):.4f}\n'
            f'Std Dev: {np.std(features):.4f}\n'
            f'Skewness: {stats.skew(features):.4f}\n'
            f'Kurtosis: {stats.kurtosis(features):.4f}\n'
            f'Max: {np.max(features):.4f}\n'
            f'Min: {np.min(features):.4f}'
        )
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        # Remove plt.show() from here
        # plt.show()  # Comment out or remove this line

        # Print detailed statistics
        print("\nFeature Vector Analysis:")
        print("-" * 30)
        print(f"Vector Shape: {features.shape}")
        print(f"Mean Value: {np.mean(features):.4f}")
        print(f"Standard Deviation: {np.std(features):.4f}")
        print(f"Skewness: {stats.skew(features):.4f}")
        print(f"Kurtosis: {stats.kurtosis(features):.4f}")
        print(f"Maximum Value: {np.max(features):.4f}")
        print(f"Minimum Value: {np.min(features):.4f}")
        print("-" * 30)

def uz():
    obj = practice()
    file_paths, titles = obj.get_file_from_user()
    if obj.image_path:
        # Now these will work in sequence
        responses = obj.generate_content()
        
        # Get doctor name and generate report
        doctor_name = input("Duty Doctor: ")
        reports = obj.process_and_generate_reports(file_paths, titles, doctor_name)
        
        # Save reports
        with open("medical_reports.txt", "w") as f:
            for report in reports:
                f.write(report + "\n\n")
        print("Reports generated and saved as 'medical_reports.txt'.")

if __name__ == "__uz__":
    uz()