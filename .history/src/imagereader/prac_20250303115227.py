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
        """Extract patient details using OCR"""
        try:
            # Initialize OCR reader
            reader = Reader(['en'])
            
            # Convert numpy array to PIL Image
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            
            # Read text from image
            results = reader.readtext(np.array(img_pil))
            
            # Initialize patient details
            patient_details = {
                "Name": "Not Found",
                "Patient ID": "Not Found",
                "Age/Sex": "Not Found",
                "Study Date": "Not Found",
                "Isotope": "Not Found",
                "Count Rate": "Not Found",
                "Administered Activity": "Not Found"
            }
            
            # Extract information from OCR results
            for detection in results:
                text = detection[1].lower()
                if "name" in text or "patient" in text:
                    patient_details["Name"] = detection[1].split(":")[-1].strip()
                elif "id" in text or "prn" in text:
                    patient_details["Patient ID"] = detection[1].split(":")[-1].strip()
                elif "age" in text or "sex" in text:
                    patient_details["Age/Sex"] = detection[1].split(":")[-1].strip()
                elif "date" in text:
                    patient_details["Study Date"] = detection[1].split(":")[-1].strip()
                elif "isotope" in text or "tc-99m" in text:
                    patient_details["Isotope"] = detection[1].strip()
                elif "count" in text or "kcnt" in text or "cnt/sec" in text:
                    patient_details["Count Rate"] = detection[1].strip()
                elif "mci" in text or "activity" in text:
                    patient_details["Administered Activity"] = detection[1].strip()
            
            # Calculate uptake value if count rate and administered activity are available
            if patient_details["Count Rate"] != "Not Found" and patient_details["Administered Activity"] != "Not Found":
                try:
                    # Extract numeric values
                    count_rate = float(''.join(filter(str.isdigit, patient_details["Count Rate"])))
                    activity = float(''.join(filter(str.isdigit, patient_details["Administered Activity"])))
                    
                    # Calculate uptake percentage (simplified formula)
                    uptake_value = (count_rate / activity) * 100
                    patient_details["Uptake Value"] = f"{uptake_value:.2f}%"
                except:
                    patient_details["Uptake Value"] = "Could not calculate"
            
            return patient_details
        
        except Exception as e:
            print(f"Error extracting patient details: {e}")
            return None

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
• Name: {patient_details["Name"]}
• Patient ID: {patient_details["Patient ID"]}
• Age/Sex: {patient_details["Age/Sex"]}
• Study Date: {patient_details["Study Date"]}
• Isotope Used: {patient_details["Isotope"]}
• Count Rate: {patient_details["Count Rate"]}
• Administered Activity: {patient_details["Administered Activity"]}
• Calculated Uptake: {patient_details.get("Uptake Value", "Not Available")}

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
        
        # Common metrics
        feature_mean = self.feature_stats['raw_mean']
        feature_std = self.feature_stats['raw_std']
        feature_kurtosis = self.feature_stats['raw_kurtosis']
        feature_skewness = self.feature_stats['raw_skewness']
        uptake_percentage = np.sum(img > 0.5) / img.size

        # Test-specific analysis
        test_parameters = {
            "THYROID": {
                "normal_uptake_range": (0.4, 0.6),
                "hot_nodule_threshold": 0.7,
                "cold_nodule_threshold": 0.3,
                "normal_distribution": (-1.0, 1.0),  # kurtosis range
                "interpretation": {
                    "hot": "Hyperfunctioning nodule/Graves disease",
                    "cold": "Possible malignancy/thyroiditis",
                    "normal": "Normal thyroid function"
                }
            },
            "BONE": {
                "normal_uptake_range": (0.45, 0.65),
                "high_uptake_threshold": 0.7,
                "low_uptake_threshold": 0.3,
                "focal_threshold": 2.0,  # kurtosis
                "interpretation": {
                    "high_focal": "Possible metastasis/osteoblastic activity",
                    "low_focal": "Possible osteolytic lesion",
                    "normal": "No significant bone pathology"
                }
            },
            "MAG3_RENAL": {
                "normal_perfusion_range": (0.45, 0.65),
                "obstruction_threshold": 0.7,
                "poor_function_threshold": 0.3,
                "time_activity": (-0.5, 0.5),  # skewness range
                "interpretation": {
                    "obstruction": "Possible renal obstruction",
                    "poor_function": "Decreased renal function",
                    "normal": "Normal renal dynamics"
                }
            },
            "DTPA": {
                "normal_gfr_range": (0.4, 0.6),
                "reduced_gfr_threshold": 0.3,
                "clearance_threshold": 0.7,
                "interpretation": {
                    "reduced": "Reduced GFR/renal insufficiency",
                    "elevated": "Increased clearance rate",
                    "normal": "Normal renal function"
                }
            },
            "DMSA": {
                "normal_uptake_range": (0.45, 0.65),
                "scarring_threshold": 0.3,
                "focal_defect_threshold": 2.0,
                "interpretation": {
                    "scarring": "Possible renal scarring",
                    "focal_defect": "Focal cortical defect",
                    "normal": "Normal cortical function"
                }
            },
            "HIDA": {
                "normal_extraction_range": (0.4, 0.6),
                "obstruction_threshold": 0.7,
                "poor_function_threshold": 0.3,
                "interpretation": {
                    "obstruction": "Possible biliary obstruction",
                    "poor_function": "Hepatocellular dysfunction",
                    "normal": "Normal hepatobiliary function"
                }
            },
            "PARATHYROID": {
                "normal_range": (0.4, 0.6),
                "adenoma_threshold": 0.7,
                "washout_threshold": -0.5,  # skewness
                "interpretation": {
                    "adenoma": "Possible parathyroid adenoma",
                    "hyperplasia": "Parathyroid hyperplasia",
                    "normal": "No abnormal parathyroid tissue"
                }
            }
        }

        if test_type not in test_parameters:
            return "Unknown test type"

        params = test_parameters[test_type]
        
        # Test-specific analysis
        analysis = f"""
🔍 {test_type} SCAN ANALYSIS
==========================

📊 QUANTITATIVE PARAMETERS
------------------------
• Mean Uptake: {feature_mean:.4f}
• Distribution (Kurtosis): {feature_kurtosis:.4f}
• Asymmetry (Skewness): {feature_skewness:.4f}
• Variability (Std): {feature_std:.4f}

🎯 INTERPRETATION
---------------
"""

        # Add test-specific interpretation
        if test_type == "THYROID":
            if feature_mean > params["hot_nodule_threshold"]:
                analysis += f"• {params['interpretation']['hot']}\n"
            elif feature_mean < params["cold_nodule_threshold"]:
                analysis += f"• {params['interpretation']['cold']}\n"
            else:
                analysis += f"• {params['interpretation']['normal']}\n"
            
        elif test_type == "BONE":
            if feature_mean > params["high_uptake_threshold"] and feature_kurtosis > params["focal_threshold"]:
                analysis += f"• {params['interpretation']['high_focal']}\n"
            elif feature_mean < params["low_uptake_threshold"]:
                analysis += f"• {params['interpretation']['low_focal']}\n"
            else:
                analysis += f"• {params['interpretation']['normal']}\n"
            
        # Add similar conditions for other test types...

        analysis += f"""
📈 TECHNICAL ASSESSMENT
--------------------
• Uptake Range: {params['normal_uptake_range']}
• Current Value: {feature_mean:.4f}
• Pattern: {'Focal' if feature_kurtosis > 2.0 else 'Diffuse'}
• Distribution: {'Symmetric' if -0.5 < feature_skewness < 0.5 else 'Asymmetric'}

⚠️ RECOMMENDATIONS
----------------
"""

        # Add test-specific recommendations
        if test_type == "THYROID":
            if feature_mean > params["hot_nodule_threshold"]:
                analysis += "• Thyroid scan with uptake measurement recommended\n"
                analysis += "• Consider thyroid function tests\n"
        # Add recommendations for other test types...

        return analysis

    def extract_scan_type(self, ocr_text):
        """Extract scan type from patient demographics"""
        scan_keywords = {
            "DMSA": ["dmsa", "renal cortex", "cortical", "kidney scan"],
            "THYROID": ["thyroid", "i-131", "tc-99m", "thyroid uptake"],
            "HIDA": ["hida", "hepatobiliary", "gallbladder", "bile", "hepatic"],
            "MAG3_RENAL": ["mag3", "renal flow", "kidney function"],
            "DTPA": ["dtpa", "gfr", "glomerular"],
            "BONE": ["bone", "skeletal", "mets", "metastases"],
            "PARATHYROID": ["parathyroid", "adenoma", "hyperplasia"]
        }
        
        text_lower = ocr_text.lower()
        for scan_type, keywords in scan_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return scan_type
        return "UNKNOWN"

    def analyze_test_parameters(self, test_type, features, img):
        """Analyze parameters based on specific test type"""
        analysis = {
            "DMSA": {
                "parameters": {
                    "Differential_Function": self.calculate_differential_function(img),
                    "Cortical_Defects": self.detect_cortical_defects(features),
                    "Scarring": self.detect_scarring(features),
                    "Kidney_Function": "Good" if np.mean(features) > 0.5 else "Reduced"
                },
                "thresholds": {
                    "normal_function": (45, 55),  # Normal differential range
                    "scarring_threshold": 0.3,
                    "cortical_defect": 0.7
                }
            },
            "THYROID": {
                "parameters": {
                    "Uptake_Percentage": self.calculate_uptake(features),
                    "Nodule_Detection": self.detect_nodules(features),
                    "Distribution": "Uniform" if stats.kurtosis(features) < 1.0 else "Non-uniform",
                    "Hot_Cold_Regions": self.detect_hot_cold_regions(features)
                },
                "thresholds": {
                    "normal_uptake": (0.5, 4.0),  # Normal uptake range %
                    "hot_nodule": 0.7,
                    "cold_nodule": 0.3
                }
            },
            "HIDA": {
                "parameters": {
                    "Ejection_Fraction": self.calculate_ejection_fraction(features),
                    "Transit_Time": self.calculate_transit_time(features),
                    "Bile_Flow": self.assess_bile_flow(features),
                    "Obstruction": self.detect_obstruction(features)
                },
                "thresholds": {
                    "normal_ef": (35, 80),  # Normal EF range %
                    "normal_transit": (45, 60),  # Minutes
                    "obstruction": 0.7
                }
            }
        }

        test_params = analysis.get(test_type, {})
        if not test_params:
            return "Test type not recognized"

        return self.generate_test_specific_report(test_type, test_params)

    def generate_test_specific_report(self, test_type, params):
        """Generate report based on test-specific parameters"""
        
        if test_type == "DMSA":
            return f"""
🔍 DMSA RENAL CORTICAL SCAN ANALYSIS
===================================

📊 QUANTITATIVE PARAMETERS
------------------------
• Differential Function: {params['parameters']['Differential_Function']}%
  (Normal Range: {params['thresholds']['normal_function'][0]}-{params['thresholds']['normal_function'][1]}%)
• Cortical Defects: {params['parameters']['Cortical_Defects']}
• Scarring Assessment: {params['parameters']['Scarring']}
• Overall Function: {params['parameters']['Kidney_Function']}

🎯 INTERPRETATION
---------------
{self.interpret_dmsa_findings(params)}
"""

        elif test_type == "THYROID":
            return f"""
🔍 THYROID SCAN ANALYSIS
======================

📊 QUANTITATIVE PARAMETERS
------------------------
• Uptake Percentage: {params['parameters']['Uptake_Percentage']}%
  (Normal Range: {params['thresholds']['normal_uptake'][0]}-{params['thresholds']['normal_uptake'][1]}%)
• Nodule Assessment: {params['parameters']['Nodule_Detection']}
• Distribution Pattern: {params['parameters']['Distribution']}
• Hot/Cold Regions: {params['parameters']['Hot_Cold_Regions']}

🎯 INTERPRETATION
---------------
{self.interpret_thyroid_findings(params)}
"""

        elif test_type == "HIDA":
            return f"""
🔍 HEPATOBILIARY SCAN ANALYSIS
============================

📊 QUANTITATIVE PARAMETERS
------------------------
• Gallbladder Ejection Fraction: {params['parameters']['Ejection_Fraction']}%
  (Normal Range: {params['thresholds']['normal_ef'][0]}-{params['thresholds']['normal_ef'][1]}%)
• Hepatic Transit Time: {params['parameters']['Transit_Time']} min
  (Normal Range: {params['thresholds']['normal_transit'][0]}-{params['thresholds']['normal_transit'][1]} min)
• Bile Flow Pattern: {params['parameters']['Bile_Flow']}
• Obstruction Assessment: {params['parameters']['Obstruction']}

🎯 INTERPRETATION
---------------
{self.interpret_hida_findings(params)}
"""

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