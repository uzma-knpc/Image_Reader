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
    report_path = os.path.join(downloads_path, f"medical_report_{doctor_name}.md")
    
    return {
        "report": reports[0],
        "grayscale": fig,
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
        gr.Markdown("# MEDICAL IMAGE ANALYSIS SYSTEM")
        gr.Markdown("## Atomic Energy Cancer Hospital, PAKISTAN")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Medical Image")
                doctor_name = gr.Textbox(label="Doctor's Name")
                process_btn = gr.Button("Process Image")
            
            with gr.Column():
                grayscale_output = gr.Plot(label="Processed Grayscale Image")
        
        report_output = gr.Textbox(label="Medical Report", lines=20)
        save_path_output = gr.Textbox(label="Save Path", visible=False)
        save_btn = gr.Button("Save Report")
        save_status = gr.Textbox(label="Save Status")
        
        # Handle processing
        outputs = process_btn.click(
            process_image,
            inputs=[input_image, doctor_name],
            outputs={
                "report": report_output,
                "grayscale": grayscale_output,
                "save_path": save_path_output
            }
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