import gradio as gr
from imagereader.prac import practice
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import shutil
from pathlib import Path

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
        "HIDA": {
            "normal_ef": (0.35, 0.8),
            "obstruction": 0.7,
            "delayed_excretion": 0.3
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
    
    else:
        return "Unknown scan type"

def process_image(image, doctor_name):
    """Process the uploaded image and generate test-specific report"""
    try:
        # Check if image is provided
        if image is None:
            return "Please upload an image first.", None, None, None
        
        # Check if doctor name is provided
        if not doctor_name:
            return "Please enter doctor's name.", None, None, None
            
        obj = practice()
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        try:
            # Save uploaded image
            image_path = "uploads/temp_image.jpg"
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                Image.fromarray(image).save(image_path)
            else:
                image.save(image_path)
            obj.image_path = image_path
            
            # Process and convert to grayscale
            img_gray = obj.load_image(image_path)
            normalized_img = obj.normalize_image(img_gray)
            
            # Extract text from image to determine scan type
            scan_type = obj.detect_scan_type(image)
            
            # Create grayscale plot
            fig_gray = plt.figure(figsize=(6, 6))
            plt.imshow(normalized_img, cmap='gray')
            plt.axis('off')
            plt.title(f"{scan_type} Analysis")
            
            # Extract features and analyze based on scan type
            tensor = obj.preprocess_image()
            features = obj.extract_features(tensor)
            
            # Generate test-specific analysis
            analysis = obj.analyze_scan_features(scan_type, features, normalized_img)
            
            # Create feature visualization with test-specific thresholds
            fig_features = create_feature_plots(features, scan_type)
            
            # Save report with test-specific name
            downloads_path = str(Path.home() / "Downloads" / "Image")
            os.makedirs(downloads_path, exist_ok=True)
            report_path = os.path.join(downloads_path, f"{scan_type}_report_{doctor_name}.txt")
            
            return analysis, fig_gray, fig_features, report_path
            
        except Exception as e:
            return f"Error processing image: {str(e)}", None, None, None
            
    except Exception as e:
        return f"Error: {str(e)}", None, None, None

def save_report(report, save_path):
    """Save the report to the specified path"""
    with open(save_path, 'w') as f:
        f.write(report)
    return f"Report saved to: {save_path}"

def main():
    # Create Gradio interface
    with gr.Blocks(title="MEDICAL IMAGE ANALYSIS SYSTEM") as iface:
        gr.Markdown("""
            <div style='text-align: center'>
                <h1 style='font-size: 2.5em; font-weight: bold'>MEDICAL IMAGE ANALYSIS SYSTEM</h1>
                <h2 style='font-size: 1.5em; font-weight: bold'>ATOMIC ENERGY CANCER HOSPITAL</h2>
                <p style='font-size: 1em; color: gray'>AI-Assisted Nuclear Medicine Analysis</p>
            </div>
        """)
        
        with gr.Row():
            with gr.Column():
                # Specify source="upload" to ensure proper image handling
                input_image = gr.Image(
                    type="numpy",
                    label="Upload Medical Image",
                    source="upload"
                )
                doctor_name = gr.Textbox(
                    label="Doctor's Name",
                    placeholder="Enter doctor's name"
                )
                process_btn = gr.Button("Process Image", variant="primary")
            
            with gr.Column():
                grayscale_output = gr.Plot(label="Processed Image")
                features_output = gr.Plot(label="Feature Analysis")
        
        report_output = gr.Textbox(label="Medical Report", lines=20)
        save_path_output = gr.Textbox(label="Save Path", visible=False)
        save_btn = gr.Button("Save Report")
        save_status = gr.Textbox(label="Save Status")
        
        # Footer
        gr.Markdown("""
            <div style='text-align: right; margin-top: 20px'>
                <p style='font-size: 12px; font-style: italic'>@Pakistan Atomic Energy Commission</p>
            </div>
        """)
        
        # Handle processing with error messages
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