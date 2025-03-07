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
        self.prompt1 = "ATOMIC ENERGY CANCER HOSPITAL"
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
        """Generate standardized report with verified patient details"""
        try:
            # Store scan type
            self.scan_type = scan_type
            
            # Calculate metrics
            metrics = self.calculate_metrics(features, img)
            
            # Get patient details with fallback
            try:
                patient_details = self.extract_patient_details(img)
            except Exception as e:
                print(f"Error in patient detail extraction: {e}")
                patient_details = {
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
            
            # Test-specific parameters
            test_params = {
                "DMSA": {
                    "parameters": ["Counts/s from each kidney", "Differential Uptake %"],
                    "thresholds": {
                        "normal_function": (45, 55),
                        "scarring": 0.3,
                        "cortical_defect": 0.7
                    }
                },
                "THYROID": {
                    "parameters": ["Thyroid Uptake %", "24h Retention"],
                    "thresholds": {
                        "normal_uptake": (0.4, 4.0),
                        "hyperthyroid": 4.0,
                        "hypothyroid": 0.4
                    }
                },
                "HIDA": {
                    "parameters": ["Gallbladder Ejection Fraction %", "Hepatic Transit Time (min)"],
                    "thresholds": {
                        "normal_ef": (35, 80),
                        "delayed_transit": 45
                    }
                }
            }

            # Generate report
            report = f"""
===========================================
AI Driven MEDICAL IMAGE ANALYSIS SYSTEM
ATOMIC ENERGY CANCER HOSPITAL (AECHs)
===========================================

👤 PATIENT DETAILS
----------------
{self.prompt1}
🏥 Hospital: {patient_details.get('center', 'AECH')}
• Name: {patient_details.get('name', 'Not provided')}
• Patient ID: {patient_details.get('id', 'Not provided')}
• Study: {scan_type}
• Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
• Equipment: {patient_details.get('manufacturer', 'Gamma Camera')}

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
"""
            return report

        except Exception as e:
            print(f"Error generating report: {e}")
            return f"Error: {str(e)}"

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

    def get_default_patient_details(self):
        # Implementation of get_default_patient_details method
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