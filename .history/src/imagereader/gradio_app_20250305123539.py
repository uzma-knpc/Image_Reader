import gradio as gr
from imagereader.prac import practice
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import shutil
from pathlib import Path
import datetime

def create_feature_plots(features, scan_type):
    """Create test-specific feature visualization"""
    fig = plt.figure(figsize=(15, 12))
    
    # Get test-specific thresholds
    thresholds = {
        "DMSA": {
            "normal_function": (0.45, 0.55),
            "scarring": 0.3,
            "cortical_defect": 0.7
        },
        "THYROID": {
            "normal_uptake": (0.4, 0.6),
            "hot_nodule": 0.7,
            "cold_nodule": 0.3
        },
        "DTPA": {
            "normal_gfr": (90, 120),
            "normal_split": (45, 55),
            "obstruction": 20  # T1/2 in minutes
        },
        "PARATHYROID": {
            "washout": 0.5,
            "retention": 0.3,
            "adenoma": 0.7
        },
        "RENAL": {
            "normal_function": (45, 55),
            "normal_tmax": (3, 5),
            "obstruction_thalf": 20
        },
        "WHOLEBODY_BONE": {
            "normal_uptake": (0.4, 0.6),
            "metastasis": 0.7,
            "metabolic": 0.3
        }
    }
    
    # Plot with test-specific reference lines
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(features, 'b-', alpha=0.6, label='Feature Values')
    if scan_type in thresholds:
        for name, value in thresholds[scan_type].items():
            if isinstance(value, tuple):
                ax1.axhline(y=value[0], color='g', linestyle='--', label=f'{name}_min')
                ax1.axhline(y=value[1], color='r', linestyle='--', label=f'{name}_max')
            else:
                ax1.axhline(y=value, color='y', linestyle='--', label=name)
    
    ax1.set_title(f'{scan_type} Feature Analysis', fontsize=14)
    ax1.legend()
    
    # Distribution histogram
    ax2 = plt.subplot(3, 1, 2)
    ax2.hist(features, bins=50, density=True)
    ax2.set_title('Feature Distribution')
    
    # Test-specific statistics
    stats_text = generate_test_specific_stats(features, scan_type)
    ax3 = plt.subplot(3, 1, 3)
    ax3.text(0.5, 0.5, stats_text, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.axis('off')
    
    plt.tight_layout()
    return fig

def generate_test_specific_stats(features, scan_type):
    """Generate statistics interpretation based on scan type"""
    mean = np.mean(features)
    std = np.std(features)
    kurtosis = stats.kurtosis(features)
    skewness = stats.skew(features)
    
    if scan_type == "DMSA":
        return f"""DMSA Scan Statistics:
• Mean Intensity: {mean:.3f} 
  → {'Normal' if 0.4 < mean < 0.6 else 'Abnormal'} kidney function
• Standard Deviation: {std:.3f}
  → {'Uniform' if std < 0.2 else 'Non-uniform'} uptake
• Kurtosis: {kurtosis:.3f}
  → {'Focal defects present' if kurtosis > 2.0 else 'No focal defects'}
• Skewness: {skewness:.3f}
  → {'Asymmetric function' if abs(skewness) > 0.5 else 'Symmetric function'}"""
    
    elif scan_type == "THYROID":
        return f"""Thyroid Scan Statistics:
• Mean Uptake: {mean:.3f}
  → {'Increased' if mean > 0.6 else 'Decreased' if mean < 0.3 else 'Normal'} uptake
• Uniformity (SD): {std:.3f}
  → {'Nodular' if std > 0.2 else 'Uniform'} distribution
• Nodule Pattern: {kurtosis:.3f}
  → {'Hot/cold nodules' if kurtosis > 2.0 else 'No significant nodules'}
• Distribution: {skewness:.3f}
  → {'Asymmetric' if abs(skewness) > 0.5 else 'Symmetric'} uptake"""
    
    elif scan_type == "HIDA":
        return f"""HIDA Scan Statistics:
• Mean Activity: {mean:.3f}
  → {'Normal' if 0.4 < mean < 0.6 else 'Abnormal'} bile flow
• Flow Variation: {std:.3f}
  → {'Obstructed' if std > 0.2 else 'Patent'} ducts
• Excretion Pattern: {kurtosis:.3f}
  → {'Delayed' if kurtosis > 2.0 else 'Normal'} excretion
• Transit Assessment: {skewness:.3f}
  → {'Abnormal' if abs(skewness) > 0.5 else 'Normal'} transit"""
    
    elif scan_type == "DTPA":
        return f"""DTPA Scan Statistics:
• Mean GFR: {mean:.3f} 
  → {'Normal' if 90 < mean < 120 else 'Abnormal'} glomerular function
• Variation: {std:.3f}
  → {'Obstructive' if std > 0.2 else 'Non-obstructive'} pattern
• Flow Pattern: {kurtosis:.3f}
  → {'Delayed excretion' if kurtosis > 2.0 else 'Normal excretion'}
• Symmetry: {skewness:.3f}
  → {'Asymmetric' if abs(skewness) > 0.5 else 'Symmetric'} function"""
    
    elif scan_type == "PARATHYROID":
        return f"""Parathyroid Scan Statistics:
• Retention: {mean:.3f}
  → {'Increased' if mean > 0.6 else 'Normal'} MIBI retention
• Distribution: {std:.3f}
  → {'Focal' if std > 0.2 else 'Diffuse'} uptake pattern
• Washout: {kurtosis:.3f}
  → {'Delayed' if kurtosis > 2.0 else 'Normal'} washout
• Symmetry: {skewness:.3f}
  → {'Asymmetric' if abs(skewness) > 0.5 else 'Symmetric'} distribution"""
    
    elif scan_type == "RENAL":
        return f"""Renal Scan Statistics:
• Function: {mean:.3f}
  → {'Normal' if 0.4 < mean < 0.6 else 'Abnormal'} renal function
• Flow Pattern: {std:.3f}
  → {'Obstructive' if std > 0.2 else 'Non-obstructive'} pattern
• Excretion: {kurtosis:.3f}
  → {'Delayed' if kurtosis > 2.0 else 'Normal'} excretion
• Symmetry: {skewness:.3f}
  → {'Asymmetric' if abs(skewness) > 0.5 else 'Symmetric'} function"""
    
    elif scan_type == "WHOLEBODY_BONE":
        return f"""Whole Body Bone Scan Statistics:
• Uptake: {mean:.3f}
  → {'Increased' if mean > 0.6 else 'Decreased' if mean < 0.3 else 'Normal'}
• Distribution: {std:.3f}
  → {'Heterogeneous' if std > 0.25 else 'Homogeneous'}
• Lesions: {kurtosis:.3f}
  → {'Multiple' if kurtosis > 2.0 else 'Single' if kurtosis > 1.5 else 'None'}
• Pattern: {skewness:.3f}
  → {'Asymmetric' if abs(skewness) > 0.5 else 'Symmetric'} distribution"""
    
    else:
        return "Unknown scan type"

def process_image(image, doctor_name):
    """Process the uploaded image and generate test-specific report"""
    try:
        if image is None:
            return "Please upload an image first.", None, None, None
        
        if not doctor_name:
            return "Please enter doctor's name.", None, None, None
            
        obj = practice()
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save uploaded image
        image_path = "uploads/temp_image.jpg"
        if isinstance(image, np.ndarray):
            Image.fromarray((image * 255).astype(np.uint8)).save(image_path)
        else:
            image.save(image_path)
        
        obj.image_path = image_path
        
        # Process image
        img_gray = obj.load_image(image_path)
        normalized_img = obj.normalize_image(img_gray)
        
        # Extract text and patient details
        extracted_text = obj.extract_text_from_image(image)
        patient_details = obj.extract_patient_details(image)
        
        # Detect scan type
        scan_type = obj.detect_scan_type(image)
        # Extract patient name and ID using OCR
        name_patient = patient_details.get('Patient name', 'Not provided')
        id_patient = patient_details.get('Patient id', 'Not provided')
        print(f"Detected scan type: {scan_type}")
        print(f"Detected name type: {name_patient}")
        print(f"Detected id type: {id_patient}")

        # Get procedure details based on scan type
        procedure_details = get_procedure_details(scan_type)
        
        # Extract features
        tensor = obj.preprocess_image()
        features = obj.extract_features(tensor)
        
        # Generate analysis
        analysis = obj.analyze_scan_features(scan_type, features, normalized_img)
        
        # Generate report with all details
        report = f"""
===========================================
AI Driven MEDICAL IMAGE ANALYSIS SYSTEM
ATOMIC ENERGY CANCER HOSPITAL (AECHs)
===========================================

📋 SCAN INFORMATION
----------------
Study Date: {patient_details.get('date_time', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}
Center: {patient_details.get('center', 'ATOMIC ENERGY MEDICAL CENTER')}
Equipment: {patient_details.get('manufacturer', 'INFINIA NUCLEAR MEDICINE')}
Study: {scan_type} Scan

💉 PROCEDURE DETAILS
-----------------
{procedure_details}

👤 PATIENT DETAILS
----------------
{json.dumps(extracted_text(), indent=2)}
• Name: {name_patient}
• Patient ID: {id_patient}
🔍  ANALYSIS
----------------
{analysis}

===========================================
REPORTING DETAILS
===========================================
Primary Report Generated by: MEDISCAN-AI
Duty Doctor: Dr. {doctor_name}
Report Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Scan Type: {scan_type}

===========================================
"""
        
        # Create visualizations
        fig_gray = plt.figure(figsize=(6, 6))
        plt.imshow(normalized_img, cmap='gray')
        plt.axis('off')
        plt.title(f"{scan_type} Scan")
        
        # Create feature plots
        fig_features = create_feature_plots(features, scan_type)
        
        # Save report to medical_reports.txt
        with open("medical_reports.txt", "w", encoding='utf-8') as f:
            f.write(report)
        
        # Save path for report
        downloads_path = str(Path.home() / "Downloads" / "Image")
        os.makedirs(downloads_path, exist_ok=True)
        report_path = os.path.join(downloads_path, f"{scan_type}_report_{doctor_name}.txt")
        
        # Save report to downloads path
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(report)
        
        return report, fig_gray, fig_features, report_path
        
    except Exception as e:
        print(f"Error in process_image: {e}")
        return f"Error processing image: {str(e)}", None, None, None

def save_report(report, save_path):
    """Save the report to the specified path"""
    with open(save_path, 'w') as f:
        f.write(report)
    return f"Report saved to: {save_path}"

def get_procedure_details(scan_type):
    """Return procedure details based on scan type"""
    procedures = {
        "DTPA": """
• Study: DTPA Renal Scan
• Radiopharmaceutical: Tc-99m DTPA
• Dose: 5-10 mCi
• Imaging: Dynamic 20-30 min
• Views: Posterior""",
        
        "PARATHYROID": """
• Study: Parathyroid Scan
• Radiopharmaceutical: Tc-99m MIBI
• Dose: 20-25 mCi
• Imaging: Early and Delayed (2-3 hrs)
• Views: Anterior neck""",
        
        "RENAL": """
• Study: Dynamic Renal Scan
• Radiopharmaceutical: Tc-99m MAG3/DTPA
• Dose: 5-10 mCi
• Imaging: Dynamic 20-30 min
• Views: Posterior""",
        
        "WHOLEBODY_BONE": """
• Study: Whole Body Bone Scan
• Radiopharmaceutical: Tc-99m MDP
• Dose: 20-25 mCi
• Imaging: 3-4 hours post injection
• Views: Anterior and Posterior whole body""",
        
        "THYROID": """
• Study: Thyroid Scan with Uptake
• Radiopharmaceutical: Tc-99m Pertechnetate/I-131
• Dose: 5-10 mCi
• Imaging: 20 minutes post injection for Tc-99m
• Views: Anterior thyroid""",
        
        "DMSA": """
• Study: DMSA Renal Cortical Scan
• Radiopharmaceutical: Tc-99m DMSA
• Dose: 3-5 mCi
• Imaging: 2-3 hours post injection
• Views: Posterior, RPO and LPO"""
    }
    
    return procedures.get(scan_type, "Procedure details not available for this scan type")

def main():
    # Create Gradio interface
    with gr.Blocks(title="MEDICAL IMAGE ANALYSIS SYSTEM") as iface:
        gr.Markdown("""
            <div style='text-align: center'>
                <h1>AI Driven MEDICAL IMAGE ANALYSIS SYSTEM (MEDISCAN-AI)</h1>
                <h2>ATOMIC ENERGY CANCER HOSPITAL (AECHs)</h2>
            </div>
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Upload Medical Image")
                doctor_name = gr.Textbox(
                    label="Duty Doctor's Name",
                    placeholder="Enter doctor's name (required)"
                )
                process_btn = gr.Button("Process Image", variant="primary")
            
            with gr.Column():
                grayscale_output = gr.Plot(label="Processed Image")
                features_output = gr.Plot(label="Feature Analysis")
        
        report_output = gr.Textbox(label="Medical Report", lines=25)
        save_path_output = gr.Textbox(label="Save Path", visible=False)
        save_btn = gr.Button("Save Report")
        save_status = gr.Textbox(label="Save Status")
        
        # Handle processing
        process_btn.click(
            process_image,
            inputs=[input_image, doctor_name],
            outputs=[report_output, grayscale_output, features_output, save_path_output]
        )
        
        # Handle saving
        save_btn.click(
            save_report,
            inputs=[report_output, save_path_output],
            outputs=save_status
        )
    
    # Launch with error handling
    try:
        iface.launch(share=True)
    except Exception as e:
        print(f"Error launching interface: {e}")

if __name__ == "__main__":
    main() 