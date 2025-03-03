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
        self.normalized_img = None
        self.image_path = None
        
        # Save model to models directory
        model_path = os.path.join("models", "fine_tuned_inception_resnet_v1.pth")
        torch.save(self.model.state_dict(), model_path)
        
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
    def preprocess_image(self):
        """Preprocess image with proper normalization"""
        if self.image_path is None:
            raise ValueError("Image path not set")
        
        # Load and convert to RGB
        image = Image.open(self.image_path).convert('RGB')
        
        # Define preprocessing with proper normalization
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # ImageNet normalization
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transform(image).unsqueeze(0)

    # Function to Extract Features
    def extract_features(self, tensor=None):
        """Extract and normalize features to 0-1 range"""
        if tensor is None:
            tensor = self.preprocess_image()
        
        with torch.no_grad():
            # Get features from the model
            features = self.model(tensor)
            features_np = features.numpy().flatten()
            
            # Normalize features to 0-1 range
            if features_np.max() - features_np.min() != 0:
                features_np = (features_np - features_np.min()) / (features_np.max() - features_np.min())
            
            print("\nFeature Statistics (After Normalization):")
            print(f"Mean: {np.mean(features_np):.4f}")
            print(f"Std Dev: {np.std(features_np):.4f}")
            print(f"Min: {np.min(features_np):.4f}")
            print(f"Max: {np.max(features_np):.4f}")
            
            return features_np

    # Function to create image embeddings
    def create_image_embedding(self, image_path=None):
        if image_path is None:
            image_path = self.image_path
        try:
            input_tensor = self.preprocess_image()
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
        """Normalize image to 0-1 range"""
        if img is None:
            return None
        # Ensure float type
        img = img.astype(float)
        # Min-max normalization
        if img.max() - img.min() != 0:
            normalized = (img - img.min()) / (img.max() - img.min())
        else:
            normalized = img
        self.normalized_img = normalized  # Store normalized image
        return normalized

    # Diagnostic function with criteria
    def diagnose_image(self, img, features):
        """Diagnose using properly scaled features"""
        # Image metrics
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        uptake_percentage = np.sum(img > 0.5) / img.size
        
        # Feature metrics with proper scaling
        feature_mean = np.mean(features)  # Should now be between 0 and 1
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
        
        # Print diagnostics
        print("\nDiagnostic Metrics:")
        print(f"Image Mean Intensity: {mean_intensity:.4f}")
        print(f"Feature Mean: {feature_mean:.4f}")
        print(f"Feature Kurtosis: {feature_kurtosis:.4f}")
        print(f"Feature Skewness: {feature_skewness:.4f}")
        
        # Determine status strings
        mean_status = "High (Hyperactive)" if feature_mean > 0.6 else "Low (Hypoactive)" if feature_mean < 0.3 else "Normal"
        kurtosis_status = "High (Focal)" if feature_kurtosis > 2.0 else "Low (Diffuse)" if feature_kurtosis < -1.0 else "Normal"
        skewness_status = "Positive (Hot Spots)" if feature_skewness > 0.5 else "Negative (Cold Spots)" if feature_skewness < -0.5 else "Normal"
        std_status = "High Variability" if feature_std > 0.25 else "Low Variability" if feature_std < 0.15 else "Normal"
        
        # Update prompt with actual values
        self.prompt = f"""This image contains a human organ image along with notes and graph.
    Analyze the image based on these measured feature values and their clinical implications:

    MEASURED FEATURE VALUES:
    1. Feature Mean: {feature_mean:.4f}
       - Normal Range: 0.3 to 0.6
       - Current Status: {mean_status}
    
    2. Feature Kurtosis: {feature_kurtosis:.4f}
       - Normal Range: -2.0 to 2.0
       - Current Status: {kurtosis_status}
    
    3. Feature Skewness: {feature_skewness:.4f}
       - Normal Range: -0.5 to 0.5
       - Current Status: {skewness_status}
    
    4. Feature Standard Deviation: {feature_std:.4f}
       - Normal Range: 0.15 to 0.25
       - Current Status: {std_status}

    Based on these specific values, provide:

    DESCRIPTION: {{
        "Organ_Type": "Describe the organ visible in image",
        "Pattern_Analysis": "Analyze uptake pattern based on measured feature mean {feature_mean:.4f}",
        "Distribution_Type": "Determine if uptake is focal or diffuse based on kurtosis {feature_kurtosis:.4f}",
        "Key_Features": "List features based on measured values"
    }}

    PREDICTION: {{
        "Primary_Condition": "Determine condition based on feature values",
        "Confidence_Level": "Assess based on feature clarity",
        "Supporting_Evidence": "Use measured feature values as evidence"
    }}

    ABNORMALITIES: {{
        "Hot_Spots": "Analyze regions with high feature values",
        "Cold_Spots": "Analyze regions with low feature values",
        "Pattern_Irregularities": "Identify based on feature distribution"
    }}

    QUANTITATIVE MEASUREMENTS: {{
        "Feature_Statistics": {{
            "Mean_Value": "{feature_mean:.4f} - {mean_status}",
            "Kurtosis": "{feature_kurtosis:.4f} - {kurtosis_status}",
            "Skewness": "{feature_skewness:.4f} - {skewness_status}",
            "Standard_Deviation": "{feature_std:.4f} - {std_status}"
        }},
        "Clinical_Significance": "Interpret these specific values in clinical context",
        "Comparison_To_Normal": "Compare measured values to normal ranges provided"
    }}

    Return a detailed analysis focusing on these specific measured values and their clinical implications.
    """
        
        # Return diagnosis based on feature values
        if (feature_mean > 0.6 and feature_kurtosis > 2.0):
            return "HIGH UPTAKE DETECTED\n- Feature analysis indicates significant abnormal patterns..."
        elif (feature_mean < 0.3 and feature_kurtosis < -1.0):
            return "LOW UPTAKE DETECTED\n- Feature analysis shows abnormal low-activity patterns..."
        else:
            return "NORMAL SCAN PATTERN\n- Feature analysis shows normal distribution..."

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
        """Generate content using both prompts with formatted response"""
        try:
            image = Image.open(self.image_path)
            
            # Generate patient details
            self.response_gen1 = self.model_gen.generate_content([self.prompt1, image])
            
            # Generate analysis
            raw_response = self.model_gen.generate_content([self.prompt, image])
            
            # Get feature values from diagnosis_metrics
            feature_mean = self.diagnosis_metrics['Feature Mean']
            feature_kurtosis = self.diagnosis_metrics['Feature Kurtosis']
            feature_skewness = self.diagnosis_metrics['Feature Skewness']
            feature_std = self.diagnosis_metrics['Feature Std']
            
            # Calculate status strings
            mean_status = ("High (Hyperactive)" if feature_mean > 0.6 
                          else "Low (Hypoactive)" if feature_mean < 0.3 
                          else "Normal")
            
            kurtosis_status = ("High (Focal)" if feature_kurtosis > 2.0 
                              else "Low (Diffuse)" if feature_kurtosis < -1.0 
                              else "Normal")
            
            skewness_status = ("Positive (Hot Spots)" if feature_skewness > 0.5 
                              else "Negative (Cold Spots)" if feature_skewness < -0.5 
                              else "Normal")
            
            std_status = ("High Variability" if feature_std > 0.25 
                         else "Low Variability" if feature_std < 0.15 
                         else "Normal")
            
            # Format the response with detailed headings
            self.response_gen = f"""
🔍 DETAILED FEATURE ANALYSIS
==========================

📊 FEATURE MEASUREMENTS
----------------------
• Feature Mean: {feature_mean:.4f}
  - Status: {mean_status}
  - Clinical Significance: Higher values indicate increased metabolic activity

• Feature Kurtosis: {feature_kurtosis:.4f}
  - Status: {kurtosis_status}
  - Pattern: {'Focal uptake patterns' if feature_kurtosis > 2.0 else 'Diffuse distribution' if feature_kurtosis < -1.0 else 'Normal distribution'}

• Feature Skewness: {feature_skewness:.4f}
  - Status: {skewness_status}
  - Indication: {'Predominant hot spots' if feature_skewness > 0.5 else 'Predominant cold spots' if feature_skewness < -0.5 else 'Balanced distribution'}

• Feature Std Dev: {feature_std:.4f}
  - Status: {std_status}
  - Variability: {'High regional variation' if feature_std > 0.25 else 'Low regional variation' if feature_std < 0.15 else 'Normal variation'}

📝 DESCRIPTION
-------------
{raw_response.text}

🎯 PREDICTION
------------
Based on feature analysis:
• Primary Finding: {
    'Hyperactive regions with focal uptake' if feature_mean > 0.6 and feature_kurtosis > 2.0
    else 'Hypoactive regions with diffuse pattern' if feature_mean < 0.3 and feature_kurtosis < -1.0
    else 'Normal uptake pattern'
}
• Pattern Type: {
    'Focal' if feature_kurtosis > 2.0
    else 'Diffuse' if feature_kurtosis < -1.0
    else 'Mixed'
}
• Distribution: {
    'Right-skewed (hot spots dominant)' if feature_skewness > 0.5
    else 'Left-skewed (cold spots dominant)' if feature_skewness < -0.5
    else 'Symmetric'
}

⚠️ ABNORMALITIES
---------------
• Hot Spots: {
    'Significant focal hot spots detected' if feature_skewness > 0.5 and feature_kurtosis > 2.0
    else 'No significant hot spots'
}
• Cold Spots: {
    'Significant cold regions present' if feature_skewness < -0.5 and feature_mean < 0.3
    else 'No significant cold spots'
}
• Pattern Analysis: {
    'Abnormal focal accumulation' if feature_kurtosis > 2.0
    else 'Abnormal diffuse pattern' if feature_kurtosis < -1.0
    else 'Normal uptake pattern'
}

📈 CLINICAL CORRELATION
---------------------
• Suggested Follow-up: {
    'Immediate clinical correlation recommended' if abs(feature_mean) > 0.6 or abs(feature_kurtosis) > 2.0
    else 'Routine follow-up advised'
}
• Confidence Level: {
    'High' if feature_std < 0.15
    else 'Medium' if feature_std < 0.25
    else 'Further investigation needed'
}
"""
        
            return self.response_gen, self.response_gen1
        except Exception as e:
            print(f"Error generating content: {e}")
            return None

    def interpret_features(self, features, img):
        """Interpret features with comprehensive analysis"""
        # Calculate all metrics
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        feature_kurtosis = stats.kurtosis(features)
        feature_skewness = stats.skew(features)
        uptake_percentage = np.sum(img > 0.5) / img.size

        # Detailed interpretation based on statistical thresholds
        analysis = f"""
🔍 DETAILED NUCLEAR SCAN ANALYSIS
===============================

📊 UPTAKE PATTERN ANALYSIS
-------------------------
1. Distribution Pattern:
   {'➤ FOCAL UPTAKE PATTERN' if feature_kurtosis > 2.0 else
    '➤ DIFFUSE UPTAKE PATTERN' if feature_kurtosis < -1.0 else
    '➤ MIXED UPTAKE PATTERN'}
   - Kurtosis Value: {feature_kurtosis:.4f}
   - {'Indicates concentrated uptake in specific regions' if feature_kurtosis > 2.0 else
      'Suggests widespread, uniform distribution' if feature_kurtosis < -1.0 else
      'Shows normal distribution pattern'}

2. Uptake Intensity:
   {'➤ HOT SPOTS DETECTED' if feature_mean > 0.6 and feature_skewness > 0.5 else
    '➤ COLD SPOTS DETECTED' if feature_mean < 0.3 and feature_skewness < -0.5 else
    '➤ NORMAL UPTAKE INTENSITY'}
   - Mean Activity: {feature_mean:.4f}
   - Skewness: {feature_skewness:.4f}
   - {'Areas of increased tracer accumulation present' if feature_mean > 0.6 else
      'Areas of decreased tracer uptake identified' if feature_mean < 0.3 else
      'Normal tracer distribution observed'}

3. Regional Distribution:
   - Symmetry: {'Asymmetric (Right dominant)' if feature_skewness > 0.5 else
                'Asymmetric (Left dominant)' if feature_skewness < -0.5 else
                'Symmetric distribution'}
   - Variability: {'High' if feature_std > 0.25 else
                   'Low' if feature_std < 0.15 else
                   'Normal'} (STD: {feature_std:.4f})

🎯 CLINICAL INTERPRETATION
------------------------
Primary Finding: {
    'Hyperactive regions with focal uptake - Suggestive of active pathology' 
        if feature_mean > 0.6 and feature_kurtosis > 2.0 else
    'Hypoactive regions with focal defects - Possible perfusion deficits' 
        if feature_mean < 0.3 and feature_kurtosis > 2.0 else
    'Diffuse hyperactivity - Consider systemic condition' 
        if feature_mean > 0.6 and feature_kurtosis < -1.0 else
    'Diffuse hypoactivity - Possible global dysfunction' 
        if feature_mean < 0.3 and feature_kurtosis < -1.0 else
    'Normal scan pattern - No significant abnormalities'
}

⚠️ SIGNIFICANT FINDINGS
---------------------
{f"• Hot spots detected in {'focal' if feature_kurtosis > 2.0 else 'diffuse'} pattern" 
    if feature_mean > 0.6 else
 f"• Cold spots detected in {'focal' if feature_kurtosis > 2.0 else 'diffuse'} pattern" 
    if feature_mean < 0.3 else
 "• No significant hot or cold spots detected"}

{f"• Asymmetric uptake with {abs(feature_skewness):.2f} skewness" 
    if abs(feature_skewness) > 0.5 else
 "• Symmetric uptake distribution"}

{f"• High variability in uptake (STD: {feature_std:.4f})" 
    if feature_std > 0.25 else ""}

📋 RECOMMENDATIONS
----------------
1. {
    'Correlate focal hot spots with anatomical imaging' 
        if feature_mean > 0.6 and feature_kurtosis > 2.0 else
    'Evaluate focal cold spots with additional imaging' 
        if feature_mean < 0.3 and feature_kurtosis > 2.0 else
    'Monitor diffuse uptake pattern' 
        if abs(feature_kurtosis) < 2.0 else
    'Regular follow-up recommended'
}

2. {
    'Consider SPECT/CT for anatomical correlation' 
        if feature_kurtosis > 2.0 else
    'Consider blood pool imaging' 
        if feature_kurtosis < -1.0 else
    'No additional imaging required at this time'
}

3. Clinical correlation advised with:
   - Laboratory findings
   - Patient symptoms
   - Prior imaging studies

📈 TECHNICAL PARAMETERS
--------------------
• Uptake Percentage: {uptake_percentage:.1%}
• Feature Variation: {feature_std:.4f}
• Distribution Shape: {'Leptokurtic (Peaked)' if feature_kurtosis > 0 else 'Platykurtic (Flat)'}
• Confidence Level: {'High' if feature_std < 0.15 else 'Moderate' if feature_std < 0.25 else 'Further views recommended'}
"""
        return analysis

    def generate_report(self, scan_id, scan_name, diagnosis, features, doctor_name):
        """Generate report using direct feature interpretation"""
        # Get direct interpretation from features
        analysis = self.interpret_features(features, self.normalized_img)
        
        report = f"""
# 🏥 MEDICAL IMAGING ANALYSIS REPORT
## Atomic Energy Cancer Hospital, PAKISTAN
### AI-Assisted Image Analysis Report

---

📋 **Report Details**
- **Report ID:** {scan_id}
- **Scan Name:** {scan_name}
- **Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{analysis}

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
        """Process images and generate reports"""
        reports = []
        rows, cols = 2, 3
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        axes = axes.flatten()

        for i, (file_path, title) in enumerate(zip(file_paths, titles)):
            if i >= len(axes):
                break

            # Load and process image
            img = self.load_image(file_path)
            self.normalized_img = self.normalize_image(img)  # Store normalized image
            
            # Extract features
            tensor = self.preprocess_image()
            features = self.extract_features(tensor)  # This will return normalized features
            
            # Generate diagnosis
            diagnosis = self.diagnose_image(self.normalized_img, features)
            
            # Generate report
            scan_id = i + 1
            report = self.generate_report(scan_id, title, diagnosis, features, doctor_name)
            reports.append(report)

            # Display image with diagnosis
            ax = axes[i]
            im = ax.imshow(self.normalized_img, cmap="gray")
            ax.set_title(f"{title}\n{diagnosis}", fontsize=10)
            ax.axis("off")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

        return reports
    
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
        responses = obj.process_and_generate_reports(file_paths, titles, "Duty Doctor")
        
        # Save reports
        with open("medical_reports.txt", "w") as f:
            for report in responses:
                f.write(report + "\n\n")
        print("Reports generated and saved as 'medical_reports.txt'.")

if __name__ == "__uz__":
    uz()