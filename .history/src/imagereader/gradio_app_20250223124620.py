import gradio as gr
from imagereader.prac import practice
from PIL import Image
import matplotlib.pyplot as plt
import os

def process_image(image, doctor_name):
    """Process the uploaded image and generate report"""
    obj = practice()
    
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Save uploaded image
    image_path = "uploads/temp_image.jpg"
    image.save(image_path)
    obj.image_path = image_path
    
    # Process image
    tensor = obj.preprocess_image()
    features = obj.extract_features(tensor)
    embeddings = obj.create_image_embedding()
    responses = obj.generate_content()
    
    # Generate report
    reports = obj.process_and_generate_reports(
        [image_path], 
        [obj.image_path], 
        doctor_name
    )
    
    return reports[0]

def main():
    # Create Gradio interface
    iface = gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(type="pil", label="Upload Medical Image"),
            gr.Textbox(label="Doctor's Name")
        ],
        outputs=gr.Textbox(label="Medical Report", lines=20),
        title="Medical Image Analysis System",
        description="Atomic Energy Cancer Hospital, PAKISTAN",
        theme="default"
    )
    iface.launch()

# Launch the interface
if __name__ == "__main__":
    main() 