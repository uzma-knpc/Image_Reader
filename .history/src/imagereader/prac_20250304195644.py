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
import re

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
        """Enhanced image normalization for better quality"""
        try:
            # Convert to float32
            img = img.astype(np.float32)
            
            # Handle RGB images
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Initial normalization to 0-255 range
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply advanced contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_clahe = clahe.apply(img_norm.astype(np.uint8))
            
            # Apply additional sharpening
            kernel = np.array([[-1,-1,-1],[-1,-1,-1]])
            img_sharp = cv2.filter2D(img_clahe, -1, kernel)
            
            # Final normalization to 0-1 range
            normalized = img_sharp.astype(np.float32) / 255.0
            
            # Store for later use
            self.normalized_img = normalized
            
            return normalized
            
        except Exception as e:
            print(f"Error in image normalization: {e}")
            return img

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
        """Enhanced patient detail extraction with better pattern matching"""
        try:
            reader = Reader(['en'], gpu=True if torch.cuda.is_available() else False)
            
            # Prepare image for OCR
            img_for_ocr = self.prepare_image_for_ocr(img)
            
            # Extract text
            results = reader.readtext(img_for_ocr)
            text = ' '.join([res[1] for res in results])
            print(f"Extracted text: {text}")  # Debug print
            
            # Enhanced pattern matching for medical scans
            patterns = {
                'name': r'(?:patient\s+name:|patient:)\s*([A-Za-z\s]+)',
                'id': r'(?:patient\s+id:|id:)\s*([A-Z0-9/-]+)',
                'study_name': r'(?:study\s+name:|scan:)\s*(\w+)',
                'date_time': r'(?:date\s+time:|date:)\s*(\d{1,2}/\d{1,2}/\d{4})',
                'center': r'(?:center:|hospital:)\s*(atomic\s+energy[^,\n]+)',
                'manufacturer': r'manufacturer\s+model:\s*([^,\n]+)',
            }
            
            details = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, text.lower(), re.IGNORECASE)
                details[key] = match.group(1).strip() if match else "Not provided"
                print(f"Found {key}: {details[key]}")  # Debug print
            
            return details
            
        except Exception as e:
            print(f"Error extracting patient details: {e}")
            return self.get_default_patient_details()

    def analyze_scan_features(self, scan_type, features, img):
        """Generate test-specific report with parameters"""
        try:
            # Get metrics and patient details
            metrics = self.calculate_metrics(features, img)
            patient_details = self.extract_patient_details(img)
            
            # Test-specific parameters
            test_params = {
                "DMSA": {
                    "parameters": ["Counts/s from each kidney", "Differential Uptake %"],
                    "thresholds": {
                        "normal_function": (45, 55),  # % range
                        "scarring": 0.3,
                        "cortical_defect": 0.7
                    }
                },
                "THYROID": {
                    "parameters": ["Thyroid Uptake %", "24h Retention"],
                    "thresholds": {
                        "normal_uptake": (0.4, 4.0),  # % range
                        "hyperthyroid": 4.0,
                        "hypothyroid": 0.4
                    }
                },
                "HIDA": {
                    "parameters": ["Gallbladder Ejection Fraction %", "Hepatic Transit Time (min)"],
                    "thresholds": {
                        "normal_ef": (35, 80),  # % range
                        "delayed_transit": 45  # minutes
                    }
                }
            }

            # Generate report with test-specific format
            report = f"""
===========================================
AI Driven MEDICAL IMAGE ANALYSIS SYSTEM
ATOMIC ENERGY CANCER HOSPITAL (AECHs)
===========================================

👤 PATIENT DETAILS
----------------
🏥 Hospital: {patient_details['center']}
• Name: {patient_details['name']}
• Patient ID: {patient_details['id']}
• Study: {patient_details['study_name']}
• Date: {patient_details['date_time']}
• Center: {patient_details['center']}
• Equipment: {patient_details['manufacturer']}

📋 PROCEDURE
-----------
• Study Type: {scan_type} Scan
• Protocol: {self.get_scan_protocol(scan_type)}
• Parameters Measured: {', '.join(test_params[scan_type]['parameters'])}

📊 ANALYSIS
----------
{self.format_test_parameters(scan_type, metrics, test_params[scan_type])}

🔍 FINDINGS
----------
{self.generate_findings(scan_type, metrics, test_params[scan_type])}

📝 DIAGNOSIS
-----------
{self.generate_diagnosis(scan_type, metrics, test_params[scan_type])}

💡 ADVICE
--------
{self.generate_recommendations(scan_type, metrics, test_params[scan_type])}

Report Generated by: MEDISCAN-AI
Validated by: Dr. {patient_details.get('referring_physician', '[Pending]')}
Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
            return report

        except Exception as e:
            print(f"Error generating report: {e}")
            return f"Error: {str(e)}"

    def format_test_parameters(self, scan_type, metrics, params):
        """Format test-specific parameters in a table-like structure"""
        if scan_type == "DMSA":
            return f"""
┌────────────────────┬─────────┬──────────┐
│ Parameter          │ Value   │ Status   │
├────────────────────┼─────────┼──────────┤
│ Left Kidney        │ {metrics['left_uptake']:.1f}%   │ {self.get_status(metrics['left_uptake'], params['thresholds']['normal_function'])} │
│ Right Kidney       │ {metrics['right_uptake']:.1f}%   │ {self.get_status(metrics['right_uptake'], params['thresholds']['normal_function'])} │
│ Differential       │ {abs(metrics['left_uptake'] - metrics['right_uptake']):.1f}%   │ {'Normal' if abs(metrics['left_uptake'] - metrics['right_uptake']) < 10 else 'Abnormal'} │
└────────────────────┴─────────┴──────────┘
"""
        elif scan_type == "THYROID":
            return f"""
┌────────────────────┬─────────┬──────────┐
│ Parameter          │ Value   │ Status   │
├────────────────────┼─────────┼──────────┤
│ Uptake (%)         │ {metrics['uptake']:.1f}%   │ {self.get_thyroid_status(metrics['uptake'])} │
│ Right Lobe         │ {metrics['right_activity']:.1f}%   │ {self.get_lobe_status(metrics['right_activity'])} │
│ Left Lobe          │ {metrics['left_activity']:.1f}%   │ {self.get_lobe_status(metrics['left_activity'])} │
└────────────────────┴─────────┴──────────┘
"""
        # Add other scan types...

    def generate_findings(self, scan_type, metrics, params):
        # Implementation of generate_findings method
        pass

    def generate_diagnosis(self, scan_type, metrics, params):
        # Implementation of generate_diagnosis method
        pass

    def generate_recommendations(self, scan_type, metrics, params):
        # Implementation of generate_recommendations method
        pass

    def get_scan_protocol(self, scan_type):
        # Implementation of get_scan_protocol method
        pass

    def get_status(self, value, threshold_range):
        # Implementation of get_status method
        pass

    def get_thyroid_status(self, uptake):
        # Implementation of get_thyroid_status method
        pass

    def get_lobe_status(self, activity):
        # Implementation of get_lobe_status method
        pass

    def prepare_image_for_ocr(self, img):
        # Implementation of prepare_image_for_ocr method
        pass

    def get_default_patient_details(self):
        # Implementation of get_default_patient_details method
        pass

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