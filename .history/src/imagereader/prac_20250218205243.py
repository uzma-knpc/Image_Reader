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
import onnxruntime as ort

from dotenv import load_dotenv

load_dotenv()
#image_path = './images/thyaemc.jpeg'
#image_path=os.path(Image_path)
class practice:
    def __init__(self):
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.eval()
        self.image_path = None  # Add this to store the path
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
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

    def get_file_from_user(self):
        """
        Get file path from user input or URL and load the file
        Returns:
            tuple: (file_paths, titles)
        """
        try:
            # Get file path or URL from user
            source = input("Enter the path to your file or URL: ")
            
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
        """Extract features from image tensor"""
        if image_tensor is None:
            # Get tensor from preprocessed image if not provided
            image_tensor = self.preprocess_image()
            print(f"image tensor{image_tensor}")
        with torch.no_grad():
            features = self.model(image_tensor)  # Extract features
        return features.numpy().flatten()

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
    def diagnose_image(self, img):
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        threshold_mean = 0.5
        threshold_std = 0.1

        # Criteria for diagnosis
        if mean_intensity > threshold_mean and std_intensity > threshold_std:
            return "Abnormal scan detected"
        else:
            return "Scan appears normal"
    # Function to calculate image metrics
    def calculate_metrics(self, img):
        return {
            "Mean Intensity": np.mean(img),
            "Standard Deviation": np.std(img),
            "Minimum Intensity": np.min(img),
            "Maximum Intensity": np.max(img)
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
    def generate_report(self, scan_id, scan_name, diagnosis, metrics, doctor_name):
        report = f"""



===================================================================================
                        NAME OF MEDICAL CENTRE, PAKISTAN
                        (AI-Assisted Image Analysis Report)
 ===================================================================================
**Report ID:** {scan_id}  
**Scan Name:** {scan_name}  
**Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
-------------------------------------------------------------------------------------
### Diagnosis
{diagnosis}
-------------------------------------------------------------------------------------
### Patient Details
{self.response_gen1.text}
-------------------------------------------------------------------------------------
### Clinical Measurements
* Mean Intensity: {metrics["Mean Intensity"]:.4f}
* Standard Deviation: {metrics["Standard Deviation"]:.4f}
* Minimum Intensity: {metrics["Minimum Intensity"]:.4f}
* Maximum Intensity: {metrics["Maximum Intensity"]:.4f}

### Analysis and Findings
{self.response_gen.text}

### Authentication
**Reporting Doctor:** {doctor_name}  
**Report Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

                                                        *Head of Department*  
                                                        Consultant  Physician  
                                                Atomic Energy Cancer Hospital, PAKISTAN
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
            diagnosis = self.diagnose_image(normalized_img)
            metrics = self.calculate_metrics(normalized_img)
            scan_id = i + 1

            # Append report
            report = self.generate_report(scan_id, title, diagnosis, metrics, doctor_name)
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
        plt.show()

        return reports  # Return the generated reports list
    
    def inference_with_onnx(self, input_tensor):
        """
        Run inference using ONNX Runtime
        Args:
            input_tensor: Preprocessed image tensor
        """
        try:
            # Get input name
            input_name = self.session.get_inputs()[0].name
            
            # Convert tensor to numpy array
            input_data = input_tensor.numpy()
            
            # Run inference
            outputs = self.session.run(None, {input_name: input_data})
            
            return outputs[0]  # Return first output
            
        except Exception as e:
            print(f"ONNX Runtime error: {e}")
            return None

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
        with open("medical_reports.md", "w") as f:
            for report in reports:
                f.write(report + "\n\n")
        print("Reports generated and saved as 'medical_reports.md'.")

if __name__ == "__uz__":
    uz()