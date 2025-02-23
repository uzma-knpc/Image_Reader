import gradio as gr
from imagereader.prac import practice
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path

def process_image(image, doctor_name):
    """Process the uploaded image and generate report"""
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
    
    # Create plot for grayscale image
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(normalized_img, cmap='gray')
    plt.axis('off')
    plt.title("Processed Grayscale Image")
    
    # Process image
    tensor = obj.preprocess_image()
    features = obj.extract_features(tensor)
    embeddings = obj.create_image_embedding()
    responses = obj.generate_content()
    
    # Generate report
    reports = obj.process_and_generate_reports(
        [image_path], 
        [os.path.basename(image_path)], 
        doctor_name
    )
    
    # Save report to Downloads/Image folder
    downloads_path = str(Path.home() / "Downloads" / "Image")
    os.makedirs(downloads_path, exist_ok=True)
    report_path = os.path.join(downloads_path, f"medical_report_{doctor_name}.txt")
    
    return reports[0], fig, report_path

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
                grayscale_output = gr.Plot(label="Processed Grayscale Image")
        
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
            outputs=[report_output, grayscale_output, save_path_output]
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