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
from easyocr import Reader
import cv2

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

        self.feature_stats = None  # Add this to store feature statistics

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
        """Extract and normalize features with proper scaling"""
        if tensor is None:
            tensor = self.preprocess_image()
        
        with torch.no_grad():
            # Get features from the model
            features = self.model(tensor)
            features_np = features.numpy().flatten()
            
            # Normalize features to 0-1 range before analysis
            normalized_features = (features_np - features_np.min()) / (features_np.max() - features_np.min())
            
            # Store normalized feature values for reporting
            self.feature_stats = {
                'raw_mean': np.mean(normalized_features),  # Now will be 0-1
                'raw_std': np.std(normalized_features),
                'raw_skewness': stats.skew(normalized_features),
                'raw_kurtosis': stats.kurtosis(normalized_features),
                'raw_min': np.min(normalized_features),
                'raw_max': np.max(normalized_features)
            }
            
            # Print normalized statistics
            print("\nNormalized Feature Statistics:")
            print(f"Mean: {self.feature_stats['raw_mean']:.4f}")  # Should be positive now
            print(f"Std Dev: {self.feature_stats['raw_std']:.4f}")
            print(f"Skewness: {self.feature_stats['raw_skewness']:.4f}")
            print(f"Kurtosis: {self.feature_stats['raw_kurtosis']:.4f}")
            print(f"Min: {self.feature_stats['raw_min']:.4f}")  # Will be 0
            print(f"Max: {self.feature_stats['raw_max']:.4f}")  # Will be 1
            
            return normalized_features  # Return normalized features

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
        """Diagnose using normalized features"""
        # Image metrics (already 0-1)
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        uptake_percentage = np.sum(img > 0.5) / img.size
        
        # Feature metrics (now also 0-1)
        feature_mean = self.feature_stats['raw_mean']  # Will be positive
        feature_std = self.feature_stats['raw_std']
        feature_kurtosis = self.feature_stats['raw_kurtosis']
        feature_skewness = self.feature_stats['raw_skewness']
        
        # Store all normalized metrics
        self.diagnosis_metrics = {
            "Mean Intensity": mean_intensity,
            "Standard Deviation": std_intensity,
            "Uptake Percentage": uptake_percentage,
            "Feature Mean": feature_mean,  # Now positive
            "Feature Std": feature_std,
            "Feature Kurtosis": feature_kurtosis,
            "Feature Skewness": feature_skewness
        }
        
        # Print diagnostic metrics for verification
        print("\nDiagnostic Metrics (All Normalized):")
        print(f"Image Mean Intensity: {mean_intensity:.4f}")
        print(f"Feature Mean: {feature_mean:.4f}")  # Should match
        print(f"Feature Kurtosis: {feature_kurtosis:.4f}")
        print(f"Feature Skewness: {feature_skewness:.4f}")
        
        # Update thresholds for normalized values
        if feature_mean > 0.6 and feature_kurtosis > 2.0:
            return "HIGH UPTAKE DETECTED\n- Feature analysis indicates significant abnormal patterns..."
        elif feature_mean < 0.3 and feature_kurtosis < -1.0:
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
        """Interpret features using consistent values"""
        # Use the stored raw feature statistics
        feature_mean = self.feature_stats['raw_mean']
        feature_std = self.feature_stats['raw_std']
        feature_kurtosis = self.feature_stats['raw_kurtosis']
        feature_skewness = self.feature_stats['raw_skewness']
        uptake_percentage = np.sum(img > 0.5) / img.size

        analysis = f"""
🔍 DETAILED NUCLEAR SCAN ANALYSIS
===============================

📊 STATISTICAL MEASUREMENTS
------------------------
• Mean Value: {feature_mean:.4f}
• Standard Deviation: {feature_std:.4f}
• Kurtosis: {feature_kurtosis:.4f}
• Skewness: {feature_skewness:.4f}
• Uptake Percentage: {uptake_percentage:.1%}

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

    def extract_patient_details(self, img):
        """Extract patient details from image using OCR"""
        try:
            reader = Reader(['en'])
            
            # Convert image to correct format for OCR
            if isinstance(img, np.ndarray):
                # Convert float64 to uint8
                if img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
                # Ensure image is grayscale
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                # Convert PIL Image to numpy array
                img = np.array(img)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Enhance image for better OCR
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply OCR
            results = reader.readtext(img)
            text = ' '.join([res[1] for res in results])
            print(f"Extracted text: {text}")  # Debug print
            
            # Extract details using patterns
            details = {
                'name': self.extract_pattern(text, r'Name:?\s*([A-Za-z\s]+)'),
                'id': self.extract_pattern(text, r'ID:?\s*([A-Z0-9-]+)'),
                'age_gender': self.extract_pattern(text, r'Age:?\s*(\d+[A-Za-z\s/]+)'),
                'referring_physician': self.extract_pattern(text, r'Dr\.?\s*([A-Za-z\s]+)'),
                'clinical_history': self.extract_pattern(text, r'History:?\s*([^\.]+)'),
                'isotope': self.extract_pattern(text, r'isotope:?\s*([A-Za-z0-9-]+)'),
                'dose': self.extract_pattern(text, r'dose:?\s*([0-9.]+\s*mCi)')
            }
            
            return details
        
        except Exception as e:
            print(f"Error extracting patient details: {e}")
            return {
                'name': 'Not provided',
                'id': 'Not provided',
                'age_gender': 'Not provided',
                'referring_physician': 'Not provided',
                'clinical_history': 'Not provided',
                'isotope': 'Not provided',
                'dose': 'Not provided'
            }

    def extract_pattern(self, text, pattern):
        """Extract information using regex pattern"""
        import re
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else "Not provided"

    def generate_report(self, scan_id, scan_name, diagnosis, features, doctor_name):
        """Generate report with patient details and uptake values"""
        # Get feature analysis
        analysis = self.interpret_features(features, self.normalized_img)
        
        # Extract patient details
        patient_details = self.extract_patient_details(self.normalized_img)
        
        report = f"""
# MEDICAL IMAGING ANALYSIS REPORT
## Atomic Energy Cancer Hospital, PAKISTAN
### AI-Assisted Image Analysis Report

---

📋 PATIENT INFORMATION
-------------------
• Name: {patient_details["name"]}
• Patient ID: {patient_details["Id"]}
• Age/Sex: {patient_details["age_gender"]}
• Study Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
• Isotope Used: {patient_details["isotope"]}
• Count Rate: {patient_details["dose"]}
• Calculated Uptake: {self.calculate_uptake(patient_details["isotope"], patient_details["dose"])}

{analysis}

✅ AUTHENTICATION
--------------
• Reporting Doctor: {doctor_name}
• Report Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
• Scan ID: {scan_id}
• Image Name: {scan_name}

---

                                                    Head of Department
                                                    Consultant Nuclear Physician
                                                    Atomic Energy Cancer Hospital
"""
        return report

    def process_and_generate_reports(self, file_paths, titles, doctor_name, test_type="THYROID"):
        """Process images with test-specific analysis"""
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
            
            # Get test-specific diagnosis
            diagnosis = self.diagnose_by_test_type(test_type, features, self.normalized_img)
            
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

    def diagnose_by_test_type(self, test_type, features, img):
        """Diagnose based on specific nuclear medicine test parameters"""
        
        # Calculate base metrics
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        feature_kurtosis = stats.kurtosis(features)
        feature_skewness = stats.skew(features)
        
        # Test-specific analysis
        if test_type == "DMSA":
            # Calculate differential function for left and right kidneys
            left_roi = img[:, :img.shape[1]//2]
            right_roi = img[:, img.shape[1]//2:]
            left_counts = np.sum(left_roi)
            right_counts = np.sum(right_roi)
            total_counts = left_counts + right_counts
            
            left_percentage = (left_counts/total_counts) * 100
            right_percentage = (right_counts/total_counts) * 100
            
            analysis = f"""
🔍 DMSA RENAL CORTICAL SCAN ANALYSIS
===================================

📊 QUANTITATIVE MEASUREMENTS
-------------------------
• Split Function:
  - Left Kidney: {left_percentage:.1f}% (Normal: 45-55%)
  - Right Kidney: {right_percentage:.1f}% (Normal: 45-55%)

• Cortical Assessment:
  - Scarring: {'Present' if feature_mean < 0.3 else 'Absent'}
  - Focal Defects: {'Detected' if feature_kurtosis > 2.0 else 'None'}
  - Cortical Thickness: {'Reduced' if feature_mean < 0.4 else 'Normal'}

🎯 INTERPRETATION
---------------
• Function: {
    'Normal bilateral function' if 45 <= left_percentage <= 55 and 45 <= right_percentage <= 55
    else 'Asymmetric function - requires correlation'
}
• Cortical Pattern: {
    'Normal cortical uptake' if feature_mean > 0.4 and feature_kurtosis < 2.0
    else 'Abnormal cortical pattern - possible scarring/defects'
}

⚠️ IMPRESSION
-----------
{self.generate_dmsa_impression(left_percentage, right_percentage, feature_mean, feature_kurtosis)}
"""

        elif test_type == "THYROID":
            # Calculate thyroid uptake and distribution
            uptake_percentage = np.sum(img > 0.5) / img.size * 100
            
            analysis = f"""
🔍 THYROID SCAN ANALYSIS
======================

📊 QUANTITATIVE MEASUREMENTS
-------------------------
• Uptake Values:
  - Total Uptake: {uptake_percentage:.1f}% (Normal: 0.5-4.0%)
  - Distribution: {'Homogeneous' if feature_std < 0.2 else 'Heterogeneous'}

• Nodule Assessment:
  - Hot Nodules: {'Present' if feature_mean > 0.7 and feature_kurtosis > 2.0 else 'Absent'}
  - Cold Nodules: {'Present' if feature_mean < 0.3 and feature_kurtosis > 2.0 else 'Absent'}
  - Pattern: {'Multinodular' if feature_kurtosis > 3.0 else 'Solitary' if feature_kurtosis > 2.0 else 'No nodules'}

🎯 INTERPRETATION
---------------
• Uptake Pattern: {
    'Increased - suggests hyperthyroidism' if uptake_percentage > 4.0
    else 'Decreased - suggests hypothyroidism' if uptake_percentage < 0.5
    else 'Normal uptake'
}
• Distribution: {
    'Toxic nodule(s) present' if feature_mean > 0.7 and feature_kurtosis > 2.0
    else 'Cold nodule(s) present' if feature_mean < 0.3 and feature_kurtosis > 2.0
    else 'Normal distribution'
}

⚠️ IMPRESSION
-----------
{self.generate_thyroid_impression(uptake_percentage, feature_mean, feature_kurtosis)}
"""

        elif test_type == "HIDA":
            # Calculate hepatobiliary parameters
            transit_time = self.calculate_transit_time(features)
            ejection_fraction = self.calculate_ef(features)
            
            analysis = f"""
🔍 HEPATOBILIARY SCAN ANALYSIS
===========================

📊 QUANTITATIVE MEASUREMENTS
-------------------------
• Time Activity Parameters:
  - Hepatic Transit Time: {transit_time:.1f} min (Normal: 45-60 min)
  - Gallbladder Ejection Fraction: {ejection_fraction:.1f}% (Normal: 35-80%)

• Flow Assessment:
  - Bile Flow: {'Normal' if feature_mean > 0.4 else 'Delayed'}
  - Obstruction: {'Present' if feature_kurtosis > 2.0 else 'Absent'}
  - Patency: {'Patent' if feature_mean > 0.3 else 'Possible obstruction'}

🎯 INTERPRETATION
---------------
• Hepatic Function: {
    'Normal hepatic extraction' if feature_mean > 0.4
    else 'Reduced hepatic function'
}
• Biliary Drainage: {
    'Normal drainage' if transit_time < 60
    else 'Delayed drainage - possible obstruction'
}
• Gallbladder Function: {
    'Normal' if 35 <= ejection_fraction <= 80
    else 'Abnormal - possible pathology'
}

⚠️ IMPRESSION
-----------
{self.generate_hida_impression(transit_time, ejection_fraction, feature_mean)}
"""

        else:
            analysis = "Unknown test type or test type not supported"
        
        return analysis

    def generate_dmsa_impression(self, left_pct, right_pct, mean, kurtosis):
        """Generate DMSA scan impression"""
        impressions = []
        
        if abs(left_pct - right_pct) > 10:
            impressions.append("Asymmetric renal function")
        if mean < 0.3:
            impressions.append("Evidence of renal scarring")
        if kurtosis > 2.0:
            impressions.append("Focal cortical defects present")
        
        if not impressions:
            return "Normal DMSA scan appearance with symmetric renal function and no evidence of scarring or cortical defects."
        
        return "IMPRESSION: " + "; ".join(impressions) + "."

    def generate_thyroid_impression(self, uptake, mean, kurtosis):
        """Generate thyroid scan impression"""
        if uptake > 4.0 and mean > 0.7:
            return "IMPRESSION: Increased thyroid uptake with hot nodule(s) - consistent with toxic nodular goiter/Graves' disease."
        elif uptake < 0.5 and mean < 0.3:
            return "IMPRESSION: Decreased thyroid uptake with cold nodule(s) - requires further evaluation to rule out malignancy."
        elif kurtosis > 2.0:
            return "IMPRESSION: Nodular thyroid with heterogeneous uptake - correlation with ultrasound recommended."
        else:
            return "IMPRESSION: Normal thyroid scan appearance with homogeneous tracer distribution."

    def generate_hida_impression(self, transit_time, ef, mean):
        """Generate HIDA scan impression"""
        impressions = []
        
        if transit_time > 60:
            impressions.append("Delayed biliary drainage")
        if ef < 35:
            impressions.append("Reduced gallbladder ejection fraction")
        if mean < 0.3:
            impressions.append("Possible biliary obstruction")
        
        if not impressions:
            return "IMPRESSION: Normal hepatobiliary scan with appropriate transit time and gallbladder ejection fraction."
        
        return "IMPRESSION: " + "; ".join(impressions) + ". Clinical correlation recommended."

    def detect_scan_type(self, image):
        """Enhanced scan type detection"""
        try:
            # Initialize OCR
            reader = Reader(['en'])
            
            # Convert image for OCR
            if isinstance(image, np.ndarray):
                image_np = image
            else:
                image_np = np.array(image)
            
            # Extract text
            results = reader.readtext(image_np)
            text = ' '.join([res[1].lower() for res in results])
            print(f"Extracted text: {text}")  # Debug print
            
            # Enhanced keywords for each scan type
            scan_keywords = {
                "BONE": ["bone", "skeletal", "whole body", "mets", "metastases", "mdp", "hmdp", "tc99m"],
                "DMSA": ["dmsa", "renal", "kidney", "cortical", "differential", "technetium", "tc-99m dmsa"],
                "THYROID": ["thyroid", "i-131", "tc-99m", "pertechnetate", "uptake", "thyroid scan"],
                "HIDA": ["hida", "hepatobiliary", "gallbladder", "liver", "biliary", "cholescintigraphy"],
                "MAG3": ["mag3", "renogram", "perfusion", "clearance", "renal function"],
                "DTPA": ["dtpa", "gfr", "glomerular", "filtration", "kidney function"],
                "PARATHYROID": ["parathyroid", "sestamibi", "adenoma", "mibi"]
            }
            
            # Check for keywords
            for scan_type, keywords in scan_keywords.items():
                if any(keyword in text for keyword in keywords):
                    print(f"Detected scan type from text: {scan_type}")  # Debug print
                    return scan_type
            
            # Fallback to feature-based detection
            if hasattr(self, 'feature_stats'):
                print("Using feature-based detection")  # Debug print
                mean = self.feature_stats['raw_mean']
                kurtosis = self.feature_stats['raw_kurtosis']
                
                # Enhanced feature-based detection
                if kurtosis > 2.0 and mean > 0.6:
                    return "THYROID"
                elif mean < 0.4 and abs(kurtosis) < 1.0:
                    return "DMSA"
                elif kurtosis > 1.5 and mean < 0.3:
                    return "HIDA"
                elif kurtosis > 2.5:
                    return "BONE"
            
            print("Warning: Using default scan type")  # Debug print
            return "BONE"  # Default to BONE if no clear match
            
        except Exception as e:
            print(f"Error in scan detection: {e}")  # Debug print
            return "UNKNOWN"

    def analyze_scan_features(self, scan_type, features, img):
        """Generate comprehensive medical report with patient details"""
        # Extract patient details using OCR
        patient_details = self.extract_patient_details(img)
        
        if scan_type == "THYROID":
            # Split image into left and right lobes
            left_lobe = img[:, :img.shape[1]//2]
            right_lobe = img[:, img.shape[1]//2:]
            
            # Calculate lobe-specific metrics
            left_uptake = np.sum(left_lobe > 0.5) / left_lobe.size * 100
            right_uptake = np.sum(right_lobe > 0.5) / right_lobe.size * 100
            
            # Calculate overall metrics
            total_uptake = np.sum(img > 0.5) / img.size * 100
            mean = np.mean(features)
            std = np.std(features)
            kurtosis = stats.kurtosis(features)
            skewness = stats.skew(features)
            
            return f"""
===========================================
AI Driven MEDICAL IMAGE ANALYSIS SYSTEM
ATOMIC ENERGY CANCER HOSPITAL (AECHs)
===========================================

👤 PATIENT DETAILS
----------------
• Name: {patient_details.get('name', 'Not provided')}
• Patient ID: {patient_details.get('id', 'Not provided')}
• Age/Gender: {patient_details.get('age_gender', 'Not provided')}
• Referral Doctor: {patient_details.get('referring_physician', 'Not provided')}
• Clinical History: {patient_details.get('clinical_history', 'Not provided')}

📋 PROCEDURE
-----------
• Study: Thyroid Scan with Uptake
• Radiopharmaceutical: {patient_details.get('isotope', 'Tc-99m/I-131')}
• Activity: {patient_details.get('dose', '5-10 mCi')}
• Imaging Time: {
    '24 hrs post administration' if 'I-131' in patient_details.get('isotope', '')
    else '20 min post administration'
}

📊 ANALYSIS
----------
• Overall Thyroid Function:
  - Total Uptake: {total_uptake:.1f}% ({
    'Increased' if total_uptake > 5 
    else 'Decreased' if total_uptake < 0.5 
    else 'Normal'})
  - Gland Size: {
    'Enlarged' if mean > 0.6 
    else 'Small' if mean < 0.3 
    else 'Normal'}

• Right Lobe Assessment:
  - Uptake: {right_uptake:.1f}% ({
    'Increased' if right_uptake > 3 
    else 'Decreased' if right_uptake < 0.3 
    else 'Normal'})
  - Size: {
    'Enlarged' if np.mean(right_lobe) > 0.6 
    else 'Small' if np.mean(right_lobe) < 0.3 
    else 'Normal'}
  - Nodules: {
    'Multiple hot nodules' if stats.kurtosis(right_lobe.flatten()) > 3.0
    else 'Single hot nodule' if stats.kurtosis(right_lobe.flatten()) > 2.0
    else 'Cold nodule(s)' if np.mean(right_lobe) < 0.3
    else 'No significant nodules'}

• Left Lobe Assessment:
  - Uptake: {left_uptake:.1f}% ({
    'Increased' if left_uptake > 3 
    else 'Decreased' if left_uptake < 0.3 
    else 'Normal'})
  - Size: {
    'Enlarged' if np.mean(left_lobe) > 0.6 
    else 'Small' if np.mean(left_lobe) < 0.3 
    else 'Normal'}
  - Nodules: {
    'Multiple hot nodules' if stats.kurtosis(left_lobe.flatten()) > 3.0
    else 'Single hot nodule' if stats.kurtosis(left_lobe.flatten()) > 2.0
    else 'Cold nodule(s)' if np.mean(left_lobe) < 0.3
    else 'No significant nodules'}

🔍 FINDINGS
----------
1. Gland Symmetry:
   - Size: {'Symmetric' if abs(np.mean(right_lobe) - np.mean(left_lobe)) < 0.1 else 'Asymmetric'}
   - Uptake: {'Symmetric' if abs(right_uptake - left_uptake) < 1.0 else 'Asymmetric'}
   - Distribution: {'Homogeneous' if std < 0.2 else 'Heterogeneous'}

2. Nodular Assessment:
   - Right Lobe: {self.describe_nodules(right_lobe)}
   - Left Lobe: {self.describe_nodules(left_lobe)}
   - Isthmus: {
    'Prominent' if np.mean(img[img.shape[0]//2-10:img.shape[0]//2+10, :]) > 0.5
    else 'Normal'}

📝 DIAGNOSIS
-----------
{self.generate_thyroid_diagnosis(total_uptake, right_uptake, left_uptake, kurtosis)}

💡 ADVICE
--------
{self.generate_thyroid_advice(total_uptake, right_uptake, left_uptake, kurtosis)}
"""

    def describe_nodules(self, lobe_img):
        """Detailed nodule description for thyroid lobes"""
        mean = np.mean(lobe_img)
        kurtosis = stats.kurtosis(lobe_img.flatten())
        hot_spots = np.sum(lobe_img > 0.7) / lobe_img.size * 100
        cold_spots = np.sum(lobe_img < 0.3) / lobe_img.size * 100
        
        if hot_spots > 5 and kurtosis > 3.0:
            return "Multiple hot nodules present"
        elif hot_spots > 2 and kurtosis > 2.0:
            return "Single hot nodule present"
        elif cold_spots > 5:
            return "Cold nodule(s) present - requires further evaluation"
        elif mean < 0.3:
            return "Hypofunctioning area present"
        else:
            return "No significant nodules"

    def generate_thyroid_diagnosis(self, total_uptake, right_uptake, left_uptake, kurtosis):
        """Generate comprehensive thyroid diagnosis"""
        diagnoses = []
        
        # Overall function
        if total_uptake > 5:
            diagnoses.append("Hyperthyroidism")
            if kurtosis > 3.0:
                diagnoses.append("Toxic multinodular goiter")
            elif kurtosis > 2.0:
                diagnoses.append("Toxic adenoma")
            else:
                diagnoses.append("Graves' disease pattern")
        elif total_uptake < 0.5:
            diagnoses.append("Hypothyroid pattern")
        
        # Asymmetry
        if abs(right_uptake - left_uptake) > 1.0:
            diagnoses.append("Asymmetric thyroid function")
        
        if not diagnoses:
            return "Normal thyroid scan appearance with symmetric uptake and no focal abnormalities."
        
        return "IMPRESSION: " + "; ".join(diagnoses) + "."

    def generate_thyroid_advice(self, total_uptake, right_uptake, left_uptake, kurtosis):
        if total_uptake > 5:
            return "Immediate clinical correlation recommended"
        elif total_uptake < 0.5:
            return "Further evaluation recommended"
        else:
            return "No immediate follow-up required"

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