import streamlit as st
from prac import practice
from PIL import Image
import matplotlib.pyplot as plt

def main():
    st.title("Medical Image Analysis System")
    st.subheader("Atomic Energy Cancer Hospital, PAKISTAN")

    # Initialize practice class
    obj = practice()

    # File upload
    uploaded_file = st.file_uploader("Choose a medical image file", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Save uploaded file
        obj.image_path = f"uploads/{uploaded_file.name}"
        image.save(obj.image_path)
        
        # Process button
        if st.button("Process Image"):
            with st.spinner('Processing...'):
                # Process image
                tensor = obj.preprocess_image()
                features = obj.extract_features(tensor)
                embeddings = obj.create_image_embedding()
                responses = obj.generate_content()
                
                # Get doctor name
                doctor_name = st.text_input("Enter Doctor's Name")
                
                if doctor_name:
                    # Generate report
                    reports = obj.process_and_generate_reports([obj.image_path], 
                                                            [uploaded_file.name], 
                                                            doctor_name)
                    
                    # Display report
                    st.markdown(reports[0])
                    
                    # Save report
                    with open("medical_reports.md", "w") as f:
                        f.write(reports[0])
                    st.success("Report generated and saved!")

if __name__ == "__main__":
    main() 