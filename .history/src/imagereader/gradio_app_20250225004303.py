import gradio as gr
from imagereader.prac import practice
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import shutil
from pathlib import Path

def create_feature_plots(features):
    """Create feature visualization plots"""
    fig = plt.figure(figsize=(15, 12))
    
    # Main feature plot
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(features, 'b-', alpha=0.6, label='Feature Values')
    ax1.plot(features, 'r.', alpha=0.5, markersize=2)
    ax1.set_title('Nuclear Image Feature Analysis', fontsize=14, pad=10)
    ax1.set_xlabel('Feature Index', fontsize=12)
    ax1.set_ylabel('Feature Value', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)

    # Distribution histogram
    ax2 = plt.subplot(3, 1, 2)
    ax2.hist(features, bins=50, color='green', alpha=0.6, density=True)
    ax2.set_title('Feature Value Distribution', fontsize=12)
    ax2.set_xlabel('Feature Value', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Statistical summary
    stats_text = (
        f'Statistical Analysis:\n'
        f'Mean: {np.mean(features):.4f}\n'
        f'Std Dev: {np.std(features):.4f}\n'
        f'Skewness: {stats.skew(features):.4f}\n'
        f'Kurtosis: {stats.kurtosis(features):.4f}\n'
        f'Max: {np.max(features):.4f}\n'
        f'Min: {np.min(features):.4f}'
    )
    
    # Add text box for statistics
    ax3 = plt.subplot(3, 1, 3)
    ax3.text(0.5, 0.5, stats_text, 
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=12)
    ax3.axis('off')

    plt.tight_layout()
    return fig

def process_image(image, doctor_name):
    """Process the uploaded image and generate report with visualizations"""
    obj = practice()
    
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Save uploaded image
    image_path = "uploads/temp_image.jpg"
    image.save(image_path)
    obj.image_path = image_path
    
    # Process and convert to grayscale
    img_gray = obj.load_image(image_path)
    normalized_img = obj.normalize_image(img_gray)
    
    # Create grayscale plot
    fig_gray = plt.figure(figsize=(6, 6))
    plt.imshow(normalized_img, cmap='gray')
    plt.axis('off')
    plt.title("Processed Grayscale Image")
    
    # Extract features and create feature plots
    tensor = obj.preprocess_image()
    features = obj.model(tensor).detach().numpy().flatten()
    fig_features = create_feature_plots(features)
    
    # Generate report
    reports = obj.process_and_generate_reports(
        [image_path], 
        [os.path.basename(image_path)], 
        doctor_name
    )
    
    # Save report
    downloads_path = str(Path.home() / "Downloads" / "Image")
    os.makedirs(downloads_path, exist_ok=True)
    report_path = os.path.join(downloads_path, f"medical_report_{doctor_name}.txt")
    
    return {
        "report": reports[0],
        "grayscale": fig_gray,
        "features": fig_features,
        "save_path": report_path
    }

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
                <p style='font-size: 1em; color: gray'>AI-Assisted Image Analysis</p>
            </div>
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", image_mode="L", label="Upload Medical Image")
                doctor_name = gr.Textbox(label="Doctor's Name")
                process_btn = gr.Button("Process Image")
            
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
    
    iface.launch(share=True)

if __name__ == "__main__":
    main() 