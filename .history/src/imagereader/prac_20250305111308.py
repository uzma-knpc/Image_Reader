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
        #self.prompt1 = "ATOMIC ENERGY CANCER HOSPITAL"
        self.feature_stats = {
            'raw_mean': 0.0,
            'raw_kurtosis': 0.0
        }

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
            
            # Apply additional sharpening with correct kernel size
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
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

    def calculate_metrics(self, features, img):
        """Calculate metrics for scan analysis"""
        try:
            # Basic statistics
            metrics = {
                'mean': np.mean(features),
                'std': np.std(features),
                'kurtosis': stats.kurtosis(features),
                'skewness': stats.skew(features)
            }
            
            # Store raw stats for scan detection
            self.feature_stats['raw_mean'] = metrics['mean']
            self.feature_stats['raw_kurtosis'] = metrics['kurtosis']
            
            # Calculate region-specific metrics
            if hasattr(self, 'scan_type'):
                if self.scan_type == "THYROID":
                    # Split image into left and right lobes
                    left_lobe = img[:, :img.shape[1]//2]
                    right_lobe = img[:, img.shape[1]//2:]
                    
                    metrics.update({
                        'uptake': self.calculate_uptake(features),
                        'left_activity': np.mean(left_lobe) * 100,
                        'right_activity': np.mean(right_lobe) * 100,
                        'symmetry': abs(np.mean(left_lobe) - np.mean(right_lobe))
                    })
                    
                elif self.scan_type == "DMSA":
                    # Calculate kidney metrics
                    left_kidney = img[:, :img.shape[1]//2]
                    right_kidney = img[:, img.shape[1]//2:]
                    
                    metrics.update({
                        'left_uptake': np.sum(left_kidney > 0.5) / left_kidney.size * 100,
                        'right_uptake': np.sum(right_kidney > 0.5) / right_kidney.size * 100,
                        'scarring': np.sum(img < 0.3) / img.size * 100
                    })
                    
                elif self.scan_type == "HIDA":
                    # Calculate hepatobiliary metrics
                    metrics.update({
                        'ef': self.calculate_ejection_fraction(features),
                        'transit_time': self.calculate_transit_time(features),
                        'obstruction': np.sum(img > 0.7) / img.size * 100
                    })
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'mean': 0.0,
                'std': 0.0,
                'kurtosis': 0.0,
                'skewness': 0.0
            }

    def calculate_uptake(self, features):
        """Calculate uptake percentage"""
        try:
            return np.sum(features > 0.5) / len(features) * 100
        except:
            return 0.0

    def calculate_ejection_fraction(self, features):
        """Calculate gallbladder ejection fraction"""
        try:
            peak = np.max(features)
            final = features[-1]
            return ((peak - final) / peak) * 100
        except:
            return 0.0

    def calculate_transit_time(self, features):
        """Calculate hepatic transit time"""
        try:
            threshold = np.max(features) * 0.5
            transit_time = np.where(features > threshold)[0][0]
            return transit_time * 0.5  # Assuming 0.5 min per frame
        except:
            return 0.0

    def analyze_scan_features(self, scan_type, features, img):
        """Generate comprehensive test-specific report"""
        try:
            # Store scan type and extracted text
            self.scan_type = scan_type
            self.extracted_text = self.extract_text_from_image(img)
            
            # Calculate metrics
            metrics = self.calculate_metrics(features, img)
            
            # Get patient details
            patient_details = self.extract_patient_details(img)

            # Test-specific content generation
            if scan_type == "DTPA":
                analysis = self.generate_dtpa_analysis(metrics)
                findings = self.generate_dtpa_findings(metrics)
                diagnosis = self.generate_dtpa_diagnosis(metrics)
                advice = self.generate_dtpa_recommendations(metrics)

            elif scan_type == "PARATHYROID":
                analysis = self.generate_parathyroid_analysis(metrics)
                findings = self.generate_parathyroid_findings(metrics)
                diagnosis = self.generate_parathyroid_diagnosis(metrics)
                advice = self.generate_parathyroid_recommendations(metrics)

            elif scan_type == "RENAL":
                analysis = self.generate_renal_analysis(metrics)
                findings = self.generate_renal_findings(metrics)
                diagnosis = self.generate_renal_diagnosis(metrics)
                advice = self.generate_renal_recommendations(metrics)

            elif scan_type == "WHOLEBODY_BONE":
                analysis = self.generate_wholebody_bone_analysis(metrics)
                findings = self.generate_wholebody_bone_findings(metrics)
                diagnosis = self.generate_wholebody_bone_diagnosis(metrics)
                advice = self.generate_wholebody_bone_recommendations(metrics)

            elif scan_type == "THYROID":
                analysis = self.generate_thyroid_analysis(metrics)
                findings = self.generate_thyroid_findings(metrics)
                diagnosis = self.generate_thyroid_diagnosis(metrics)
                advice = self.generate_thyroid_recommendations(metrics)

            elif scan_type == "DMSA":
                analysis = self.generate_dmsa_analysis(metrics)
                findings = self.generate_dmsa_findings(metrics)
                diagnosis = self.generate_dmsa_diagnosis(metrics)
                advice = self.generate_dmsa_recommendations(metrics)

            else:
                return f"Unsupported scan type: {scan_type}"

            # Format the complete report
            report = f"""
📊 ANALYSIS
----------
{analysis}

🔍 FINDINGS
----------
{findings}

📝 DIAGNOSIS
-----------
{diagnosis}

💡 ADVICE
--------
{advice}
"""
            return report

        except Exception as e:
            print(f"Error generating report: {e}")
            return str(e)

    def extract_text_from_image(self, img):
        """Extract all text from image"""
        try:
            reader = Reader(['en'])
            results = reader.readtext(np.array(img))
            return ' '.join([res[1] for res in results])
        except Exception as e:
            print(f"Error extracting text: {e}")
            return "Text extraction failed"

    def get_default_patient_details(self):
        """Return default patient details with example values"""
        return {
            'name': 'MUHAMMAD HASNAIN',
            'id': '002421/25',
            'age_gender': '1.1 YEARS',
            'study_name': 'RENAL HEAD',
            'date_time': '2/22/2025',
            'center': 'ATOMIC ENERGY MEDICAL CENTER',
            'manufacturer': 'INFINIA NUCLEAR MEDICINE',
            'pediatric_state': 'YES',
            'isotope': 'Tc-MAG3',
            'diuretic': 'YES',
            'diuretic_time': '15 MIN',
            'duty_doctor': 'ON-CALL NUCLEAR PHYSICIAN',
            'referring_physician': 'PENDING VALIDATION'
        }

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
Paitient Details:
{self.extract_patient_details(self.image_path)}
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
        """Enhanced patient detail extraction with better image handling"""
        try:
            reader = Reader(['en'], gpu=True if torch.cuda.is_available() else False)
            
            # Convert image to proper format for OCR
            if isinstance(img, np.ndarray):
                if img.dtype == np.float64 or img.dtype == np.float32:
                    img_for_ocr = (img * 255).astype(np.uint8)
                else:
                    img_for_ocr = img.astype(np.uint8)
                    
                # Ensure image is grayscale
                if len(img_for_ocr.shape) == 3:
                    img_for_ocr = cv2.cvtColor(img_for_ocr, cv2.COLOR_RGB2GRAY)
                    
                # Convert to PIL Image for OCR
                img_for_ocr = Image.fromarray(img_for_ocr)
            else:
                img_for_ocr = img
            
            # Extract text
            results = reader.readtext(np.array(img_for_ocr))
            text = ' '.join([res[1] for res in results])
            print(f"Extracted text: {text}")  # Debug print
            
            # Default details in case extraction fails
            details = {
                'name': 'Not provided',
                'id': 'Not provided',
                'age_gender': 'Not provided',
                'referring_physician': 'Not provided',
                'clinical_history': 'Not provided',
                'isotope': 'Not provided',
                'dose': 'Not provided',
                'center': 'AECH',
                'manufacturer': 'Gamma Camera'
            }
            
            # Enhanced pattern matching
            patterns = {
                'name': r'(?:Name|Patient)[:\s]+([A-Za-z\s]+)',
                'id': r'(?:ID|MR)[:\s]*([A-Z0-9-/]+)',
                'age_gender': r'(?:Age|Sex)[:\s]*(\d+\s*[YyMm](?:rs)?\.?\s*[/\s]+[MmFf])',
                'referring_physician': r'(?:Dr|Doctor|Ref)[:\s]+([A-Za-z\s.]+)',
                'clinical_history': r'(?:History|Clinical)[:\s]+([^\.]+)',
                'isotope': r'(?:Isotope|Radio)[:\s]+([A-Za-z0-9-]+)',
                'dose': r'(?:Dose|Activity)[:\s]+(\d+\.?\d*\s*mCi)',
                'center': r'(?:center:|hospital:)\s*(atomic\s+energy[^,\n]+)',
                'manufacturer': r'manufacturer\s+model:\s*([^,\n]+)'
            }
            
            # Update details with any found matches
            for key, pattern in patterns.items():
                match = re.search(pattern, text.lower(), re.IGNORECASE)
                if match:
                    details[key] = match.group(1).strip()
                    print(f"Found {key}: {details[key]}")  # Debug print
            
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
                'dose': 'Not provided',
                'center': 'AECH',
                'manufacturer': 'Gamma Camera'
            }

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
        """Generate findings based on scan type and metrics"""
        try:
            if scan_type == "THYROID":
                return self.generate_thyroid_findings(metrics, params)
            elif scan_type == "DMSA":
                return self.generate_dmsa_findings(metrics, params)
            elif scan_type == "HIDA":
                return self.generate_hida_findings(metrics, params)
            else:
                return "No specific findings available for this scan type."
        except Exception as e:
            print(f"Error generating findings: {e}")
            return "Error generating findings"

    def generate_diagnosis(self, scan_type, metrics, params):
        """Generate diagnosis based on scan type and metrics"""
        try:
            if scan_type == "THYROID":
                return self.generate_thyroid_impression(metrics['mean'], metrics['std'], 
                                                     metrics['kurtosis'], metrics['skewness'])
            elif scan_type == "DMSA":
                return self.generate_dmsa_impression(metrics)
            elif scan_type == "HIDA":
                return self.generate_hida_impression(metrics)
            else:
                return "No specific diagnosis available for this scan type."
        except Exception as e:
            print(f"Error generating diagnosis: {e}")
            return "Error generating diagnosis"

    def generate_recommendations(self, scan_type, metrics, params):
        """Generate recommendations based on scan type and metrics"""
        try:
            recommendations = []
            if scan_type == "THYROID":
                if metrics['mean'] > params['thresholds']['hyperthyroid']:
                    recommendations.append("1. Thyroid function tests (TSH, FT4, FT3)")
                    recommendations.append("2. Nuclear Medicine consultation for therapy")
                elif metrics['mean'] < params['thresholds']['hypothyroid']:
                    recommendations.append("1. Thyroid function tests")
                    recommendations.append("2. Endocrine consultation")
            elif scan_type == "DMSA":
                if metrics.get('scarring', 0) > params['thresholds']['scarring']:
                    recommendations.append("1. Correlation with ultrasound")
                    recommendations.append("2. Urology consultation")
            
            if not recommendations:
                recommendations.append("Routine follow-up as clinically indicated")
            
            return "\n".join(recommendations)
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return "Follow-up as clinically indicated"

    def get_scan_protocol(self, scan_type):
        """Return protocol based on scan type"""
        protocols = {
            "THYROID": "Static anterior images at 20 minutes post injection",
            "DMSA": "Static posterior images at 3 hours post injection",
            "HIDA": "Dynamic images at 1 min/frame for 60 minutes",
            "BONE": "Whole body anterior and posterior images at 3 hours",
        }
        return protocols.get(scan_type, "Standard protocol")

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

    def detect_scan_type(self, image):
        """Detect scan type from image using OCR and keywords"""
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
            print(f"Extracted text for scan detection: {text}")  # Debug print
            
            # Define keywords for each scan type
            scan_keywords = {
                "DMSA": ["dmsa", "renal", "kidney", "cortical", "differential", "technetium", "tc-99m dmsa"],
                "THYROID": ["thyroid", "i-131", "tc-99m", "pertechnetate", "uptake", "thyroid scan"],
                "HIDA": ["hida", "hepatobiliary", "gallbladder", "liver", "biliary", "cholescintigraphy"],
                "MAG3": ["mag3", "renogram", "perfusion", "clearance", "renal function"],
                "DTPA": ["dtpa", "gfr", "glomerular", "filtration", "kidney function"],
                "BONE": ["bone", "skeletal", "whole body", "mets", "metastases", "mdp"],
                "PARATHYROID": ["parathyroid", "sestamibi", "adenoma", "mibi"]
            }
            
            # Check for keywords in the extracted text
            for scan_type, keywords in scan_keywords.items():
                if any(keyword in text for keyword in keywords):
                    print(f"Detected scan type: {scan_type}")  # Debug print
                    return scan_type
            
            # Fallback to feature-based detection
            if hasattr(self, 'feature_stats'):
                print("Using feature-based detection")  # Debug print
                mean = self.feature_stats['raw_mean']
                kurtosis = self.feature_stats['raw_kurtosis']
                
                if kurtosis > 2.0 and mean > 0.6:
                    return "THYROID"
                elif mean < 0.4 and abs(kurtosis) < 1.0:
                    return "DMSA"
                elif kurtosis > 1.5 and mean < 0.3:
                    return "HIDA"
                elif kurtosis > 2.5:
                    return "BONE"
            
            print("Warning: Using default scan type")  # Debug print
            return "THYROID"  # Default to thyroid if no clear match
            
        except Exception as e:
            print(f"Error in scan detection: {e}")  # Debug print
            return "UNKNOWN"

    def generate_bone_analysis(self, metrics):
        """Generate bone scan analysis"""
        return f"""
1. Quantitative Parameters:
   • Overall Activity: {metrics.get('mean', 0):.2f} ({'High' if metrics.get('mean', 0) > 0.6 else 'Low' if metrics.get('mean', 0) < 0.3 else 'Normal'})
   • Distribution: {metrics.get('std', 0):.2f} ({'Heterogeneous' if metrics.get('std', 0) > 0.25 else 'Homogeneous'})
   • Focal Areas: {metrics.get('kurtosis', 0):.2f} ({'Multiple' if metrics.get('kurtosis', 0) > 2.0 else 'Single' if metrics.get('kurtosis', 0) > 1.5 else 'None'})
   • Symmetry: {metrics.get('skewness', 0):.2f} ({'Asymmetric' if abs(metrics.get('skewness', 0)) > 0.5 else 'Symmetric'})

2. Regional Assessment:
   • Axial Skeleton: {'Abnormal' if metrics.get('mean', 0) > 0.6 else 'Normal'} uptake
   • Appendicular Skeleton: {'Abnormal' if metrics.get('std', 0) > 0.25 else 'Normal'} distribution
   • Focal Lesions: {
       'Multiple lesions present' if metrics.get('kurtosis', 0) > 2.0
       else 'Single lesion present' if metrics.get('kurtosis', 0) > 1.5
       else 'No significant focal lesions'
   }

3. Pattern Analysis:
   • Distribution Type: {'Diffuse' if metrics.get('std', 0) < 0.25 else 'Focal'}
   • Intensity: {'Intense' if metrics.get('mean', 0) > 0.7 
                else 'Moderate' if metrics.get('mean', 0) > 0.4 
                else 'Mild'}
   • Symmetry: {'Symmetric' if abs(metrics.get('skewness', 0)) < 0.5 
               else 'Asymmetric with ' + ('right' if metrics.get('skewness', 0) > 0 else 'left') + ' predominance'}"""

    def generate_bone_findings(self, metrics):
        """Generate bone scan findings"""
        findings = []
        mean = metrics.get('mean', 0)
        std = metrics.get('std', 0)
        kurtosis = metrics.get('kurtosis', 0)
        skewness = metrics.get('skewness', 0)
        
        # Overall uptake pattern
        if mean > 0.6:
            findings.append("• Increased tracer uptake noted")
        elif mean < 0.3:
            findings.append("• Reduced tracer uptake noted")
        else:
            findings.append("• Normal tracer distribution")
        
        # Distribution pattern
        if std > 0.25:
            findings.append("• Heterogeneous distribution pattern")
        
        # Focal lesions
        if kurtosis > 2.0:
            findings.append("• Multiple focal lesions identified")
        elif kurtosis > 1.5:
            findings.append("• Single focal lesion identified")
        
        # Symmetry
        if abs(skewness) > 0.5:
            findings.append(f"• Asymmetric uptake with {'right' if skewness > 0 else 'left'} side predominance")
        
        if not findings:
            findings.append("• No significant abnormality detected")
        
        return "\n".join(findings)

    def generate_bone_diagnosis(self, metrics):
        """Generate bone scan diagnosis"""
        mean = metrics.get('mean', 0)
        std = metrics.get('std', 0)
        kurtosis = metrics.get('kurtosis', 0)
        
        # Multiple conditions check
        conditions = []
        
        if mean > 0.6:
            if kurtosis > 2.0:
                conditions.append("Multiple focal lesions suggestive of metastatic disease")
            elif kurtosis > 1.5:
                conditions.append("Focal increased uptake - requires correlation")
            else:
                conditions.append("Diffusely increased skeletal uptake")
            
        if std > 0.25:
            conditions.append("Heterogeneous distribution pattern")
        
        if metrics.get('skewness', 0) > 0.5:
            conditions.append("Asymmetric uptake pattern")
        
        if not conditions:
            return "IMPRESSION: Normal bone scan without evidence of metastatic disease"
        
        return "IMPRESSION: " + "; ".join(conditions)

    def generate_bone_recommendations(self, metrics):
        """Generate bone scan recommendations"""
        recommendations = []
        mean = metrics.get('mean', 0)
        kurtosis = metrics.get('kurtosis', 0)
        std = metrics.get('std', 0)
        
        # High uptake with multiple lesions
        if mean > 0.6 and kurtosis > 2.0:
            recommendations.extend([
                "1. Correlation with tumor markers",
                "2. CT/MRI of areas showing increased uptake",
                "3. Early oncology consultation",
                "4. Follow-up scan in 3-4 months after treatment"
            ])
        
        # Single focal lesion
        elif kurtosis > 1.5:
            recommendations.extend([
                "1. Radiographic correlation of the focal area",
                "2. Consider CT/MRI for anatomical detail",
                "3. Follow-up scan in 6 months"
            ])
        
        # Diffuse changes
        elif mean > 0.6 or std > 0.25:
            recommendations.extend([
                "1. Biochemical markers (ALP, Ca, P)",
                "2. Consider metabolic bone disease workup",
                "3. Follow-up scan based on clinical correlation"
            ])
        
        # Normal scan
        else:
            recommendations.append("Routine follow-up as clinically indicated")
        
        return "\n".join(recommendations)

    def generate_thyroid_analysis(self, metrics):
        """Generate thyroid scan analysis"""
        return f"""
1. Quantitative Parameters:
   • Mean Uptake: {metrics.get('mean', 0):.2f}% ({'High' if metrics.get('mean', 0) > 0.6 else 'Low' if metrics.get('mean', 0) < 0.3 else 'Normal'})
   • Uniformity: {metrics.get('std', 0):.2f} ({'Heterogeneous' if metrics.get('std', 0) > 0.25 else 'Homogeneous'})
   • Nodularity: {metrics.get('kurtosis', 0):.2f} ({'Multiple' if metrics.get('kurtosis', 0) > 2.0 else 'Single' if metrics.get('kurtosis', 0) > 1.5 else 'None'})
   • Symmetry: {metrics.get('skewness', 0):.2f} ({'Asymmetric' if abs(metrics.get('skewness', 0)) > 0.5 else 'Symmetric'})

2. Lobe-Specific Analysis:
   • Right Lobe: {metrics.get('right_activity', 0):.1f}% uptake
   • Left Lobe: {metrics.get('left_activity', 0):.1f}% uptake
   • Ratio (R/L): {metrics.get('right_activity', 0)/metrics.get('left_activity', 1):.2f}"""

    def generate_thyroid_findings(self, metrics):
        """Generate thyroid scan findings"""
        findings = []
        mean = metrics.get('mean', 0)
        std = metrics.get('std', 0)
        kurtosis = metrics.get('kurtosis', 0)
        
        if mean > 0.6:
            findings.append("• Increased thyroid uptake")
        elif mean < 0.3:
            findings.append("• Decreased thyroid uptake")
        else:
            findings.append("• Normal thyroid uptake")
        
        if std > 0.25:
            findings.append("• Heterogeneous distribution")
        if kurtosis > 2.0:
            findings.append("• Multiple focal areas noted")
        elif kurtosis > 1.5:
            findings.append("• Single focal area noted")
        
        return "\n".join(findings) if findings else "• No significant abnormality detected"

    def generate_thyroid_diagnosis(self, metrics):
        """Generate thyroid scan diagnosis"""
        mean = metrics.get('mean', 0)
        kurtosis = metrics.get('kurtosis', 0)
        
        if mean > 0.6:
            if kurtosis > 2.0:
                return "IMPRESSION: Features consistent with toxic multinodular goiter"
            elif kurtosis > 1.5:
                return "IMPRESSION: Features suggestive of toxic adenoma"
            else:
                return "IMPRESSION: Features consistent with Graves' disease"
        elif mean < 0.3:
            return "IMPRESSION: Reduced thyroid function - suggest thyroid function tests"
        else:
            return "IMPRESSION: Normal thyroid scan"

    def generate_thyroid_recommendations(self, metrics):
        """Generate thyroid scan recommendations"""
        mean = metrics.get('mean', 0)
        kurtosis = metrics.get('kurtosis', 0)
        recommendations = []
        
        if mean > 0.6:
            recommendations.extend([
                "1. Thyroid function tests (TSH, FT4, FT3)",
                "2. Nuclear Medicine consultation for radioiodine therapy planning",
                "3. Follow-up scan in 1-2 months after treatment"
            ])
        elif mean < 0.3:
            recommendations.extend([
                "1. Complete thyroid profile",
                "2. Endocrine consultation",
                "3. Follow-up scan in 3-6 months"
            ])
        
        if kurtosis > 1.5:
            recommendations.append("4. Correlation with thyroid ultrasound recommended")
        
        return "\n".join(recommendations) if recommendations else "Routine follow-up as clinically indicated"

    def generate_dmsa_analysis(self, metrics):
        """Generate DMSA scan analysis"""
        return f"""
1. Quantitative Parameters:
   • Left Kidney: {metrics.get('left_uptake', 0):.1f}% uptake
   • Right Kidney: {metrics.get('right_uptake', 0):.1f}% uptake
   • Differential Function: Left {metrics.get('left_uptake', 0):.1f}% : Right {metrics.get('right_uptake', 0):.1f}%
   • Cortical Defects: {'Present' if metrics.get('std', 0) > 0.25 else 'Absent'}"""

    def generate_dmsa_findings(self, metrics):
        """Generate DMSA scan findings"""
        findings = []
        left = metrics.get('left_uptake', 0)
        right = metrics.get('right_uptake', 0)
        
        if abs(left - right) > 10:
            findings.append(f"• Asymmetric function with {'left' if left > right else 'right'} predominance")
        if metrics.get('std', 0) > 0.25:
            findings.append("• Cortical defects noted")
        
        return "\n".join(findings) if findings else "• Normal cortical function bilaterally"

    def generate_dmsa_diagnosis(self, metrics):
        """Generate DMSA scan diagnosis"""
        left = metrics.get('left_uptake', 0)
        right = metrics.get('right_uptake', 0)
        
        if abs(left - right) > 10:
            return f"IMPRESSION: Asymmetric renal function with {('left' if left > right else 'right')} kidney predominance"
        return "IMPRESSION: Normal bilateral renal cortical function"

    def generate_dmsa_recommendations(self, metrics):
        """Generate DMSA scan recommendations"""
        recommendations = []
        if abs(metrics.get('left_uptake', 0) - metrics.get('right_uptake', 0)) > 10:
            recommendations.extend([
                "1. Correlation with renal function tests",
                "2. Urology consultation advised",
                "3. Follow-up scan in 6 months"
            ])
        if metrics.get('std', 0) > 0.25:
            recommendations.append("4. Consider CT/MRI for anatomical correlation")
        
        return "\n".join(recommendations) if recommendations else "Routine follow-up as clinically indicated"

    def generate_dtpa_analysis(self, metrics):
        """Generate DTPA scan analysis"""
        return f"""
1. Quantitative Parameters:
   • GFR: {metrics.get('gfr', 0):.1f} ml/min
   • Split Function: Left {metrics.get('left_function', 0):.1f}% : Right {metrics.get('right_function', 0):.1f}%
   • T-max: Left {metrics.get('left_tmax', 0):.1f} min : Right {metrics.get('right_tmax', 0):.1f} min
   • T1/2: Left {metrics.get('left_thalf', 0):.1f} min : Right {metrics.get('right_thalf', 0):.1f} min

2. Flow Assessment:
   • Perfusion: {'Normal' if metrics.get('mean', 0) > 0.4 else 'Reduced'}
   • Pattern: {'Obstructive' if metrics.get('kurtosis', 0) > 2.0 else 'Non-obstructive'}"""

    def generate_parathyroid_analysis(self, metrics):
        """Generate parathyroid scan analysis"""
        return f"""
1. Quantitative Parameters:
   • Early Phase Uptake: {metrics.get('early_uptake', 0):.1f}%
   • Delayed Phase Uptake: {metrics.get('delayed_uptake', 0):.1f}%
   • Washout Rate: {metrics.get('washout_rate', 0):.1f}%
   • Retention Index: {metrics.get('retention_index', 0):.1f}

2. Regional Assessment:
   • Thyroid Pattern: {'Nodular' if metrics.get('std', 0) > 0.25 else 'Uniform'}
   • Focal Areas: {'Present' if metrics.get('kurtosis', 0) > 2.0 else 'Absent'}
   • Distribution: {'Asymmetric' if abs(metrics.get('skewness', 0)) > 0.5 else 'Symmetric'}"""

    def generate_renal_analysis(self, metrics):
        """Generate renal scan analysis"""
        return f"""
1. Quantitative Parameters:
   • Left Kidney Function: {metrics.get('left_function', 0):.1f}%
   • Right Kidney Function: {metrics.get('right_function', 0):.1f}%
   • Left T-max: {metrics.get('left_tmax', 0):.1f} min
   • Right T-max: {metrics.get('right_tmax', 0):.1f} min
   • Left T1/2: {metrics.get('left_thalf', 0):.1f} min
   • Right T1/2: {metrics.get('right_thalf', 0):.1f} min

2. Flow Assessment:
   • Perfusion Phase: {'Normal' if metrics.get('mean', 0) > 0.4 else 'Reduced'}
   • Excretion Pattern: {'Obstructive' if metrics.get('kurtosis', 0) > 2.0 else 'Non-obstructive'}"""

    def generate_wholebody_bone_analysis(self, metrics):
        """Generate whole body bone scan analysis"""
        return f"""
1. Quantitative Parameters:
   • Overall Activity: {metrics.get('mean', 0):.2f} ({'High' if metrics.get('mean', 0) > 0.6 else 'Low' if metrics.get('mean', 0) < 0.3 else 'Normal'})
   • Distribution: {metrics.get('std', 0):.2f} ({'Heterogeneous' if metrics.get('std', 0) > 0.25 else 'Homogeneous'})
   • Focal Areas: {metrics.get('kurtosis', 0):.2f} ({'Multiple' if metrics.get('kurtosis', 0) > 2.0 else 'Single' if metrics.get('kurtosis', 0) > 1.5 else 'None'})
   • Symmetry: {metrics.get('skewness', 0):.2f} ({'Asymmetric' if abs(metrics.get('skewness', 0)) > 0.5 else 'Symmetric'})

2. Regional Assessment:
   • Axial Skeleton: {'Abnormal' if metrics.get('mean', 0) > 0.6 else 'Normal'} uptake
   • Appendicular Skeleton: {'Abnormal' if metrics.get('std', 0) > 0.25 else 'Normal'} distribution
   • Focal Lesions: {
       'Multiple lesions present' if metrics.get('kurtosis', 0) > 2.0
       else 'Single lesion present' if metrics.get('kurtosis', 0) > 1.5
       else 'No significant focal lesions'
   }"""

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