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
        self.prompt1 = """This image contains a human organ image  along with some notes and graph
            Given the Medical image, "Analyze the given medical image and extract
            the patient details from image and return the results in formated style.

        Patient_Detail: [
        "Name",
        "PRN-NO",
        "Isotop",
        "Scan Date",
        ...
    ]
    }
            """
        self.prompt = """This image contains a human organ image  along with some notes and graph
            Given the Medical image, describe the parameters  thoroughly as possible based on what you
            see in the image, "Analyze the given medical image, which includes a human organ along with notes and a graph. Provide a detailed description of the organ, including its features, intensity, and any observed parameters.
            Based on the image, by this analysis a conclusive advice given to Patient  and return the results in JSON format.

        DESCRIPTION: "\n<Provide a detailed description of the organ>\n",
        PRIDICTION: "\n<Provide a provisional or confirmed diagnosis>\n,
        ABNORMALITIES:<Provide a detailed description of the organ>",

        QAUNTITATIVE MEASUREMENTS:[
        "Organ size",
        "Region of Intrest",
        "Intensity",
        "Number of Nodules",
        "standard uptake value",
        "Type of Radiolabeled substance",
        ...
        ]
        Organ Features: [
        "feature1",
        "feature2",
        "feature3",
        "feature4",
        ...
        ]

    }

            """
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
        """Diagnose image based primarily on feature metrics with image metrics as supporting evidence"""
        # Image metrics (supporting evidence)
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        uptake_percentage = np.sum(img > 0.5) / img.size
        
        # Feature metrics (primary decision criteria)
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        feature_kurtosis = stats.kurtosis(features)
        feature_skewness = stats.skew(features)
        
        # Store metrics
        self.diagnosis_metrics = {
            # Supporting metrics
            "Mean Intensity": mean_intensity,
            "Standard Deviation": std_intensity,
            "Uptake Percentage": uptake_percentage,
            
            # Primary decision metrics
            "Feature Mean": feature_mean,
            "Feature Std": feature_std,
            "Feature Kurtosis": feature_kurtosis,
            "Feature Skewness": feature_skewness
        }
        
        # Primary decision based on feature metrics
        # with confirmation from image metrics
        if (feature_mean > 0.6 and feature_kurtosis > 2.0 and  # Primary criteria
            mean_intensity > 0.5 and uptake_percentage > 0.4):  # Supporting criteria
            return """HIGH UPTAKE DETECTED
            - Feature analysis indicates significant abnormal patterns
            - Direct image metrics confirm high uptake regions
            - Findings suggest hyperfunctioning areas (hot spots)
            - Potential conditions: hyperthyroidism, bone metastases, renal obstruction
            - Further clinical correlation recommended
            """
            
        elif (feature_mean < 0.3 and feature_kurtosis < -1.0 and  # Primary criteria
              mean_intensity < 0.3 and uptake_percentage < 0.2):  # Supporting criteria
            return """LOW UPTAKE DETECTED
            - Feature analysis shows abnormal low-activity patterns
            - Direct image metrics confirm low uptake regions
            - Findings suggest hypo-functioning areas (cold spots)
            - Potential conditions: nodules, renal dysfunction, bone metastases
            - Clinical correlation advised
            """
            
        else:
            return """NORMAL SCAN PATTERN
            - Feature analysis shows normal distribution
            - Direct image metrics within normal range
            - No significant abnormalities detected
            - Regular follow-up recommended
            """

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

    def generate_content(self, image_path=None):
        """Generate content using Gemini model"""
        if image_path is None:
            image_path = self.image_path
        image = Image.open(image_path)
        self.response_gen = self.model_gen.generate_content([self.prompt, image])
        self.response_gen1 = self.model_gen.generate_content([self.prompt1, image])
        return self.response_gen, self.response_gen1

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
        tensor = obj.preprocess_image()
        features = obj.extract_features(tensor)
        embeddings = obj.create_image_embedding()
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