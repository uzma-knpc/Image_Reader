
import google.generativeai as genai
import os

import shutil
import requests

import datetime
from PIL import Image
from transformers import pipeline
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from dotenv import load_dotenv
load_dotenv()
#image_path = './images/thyaemc.jpeg'
#image_path=os.path(Image_path)
class practice():
    #image_path= input("Enter the full path to the image file: ")
    
    #torch.save(model.state_dict(), "fine_tuned_inception_resnet_v1.pth")

    def get_file_from_user():
        """
    Get file path from user input and load the file
    Returns:
        str: Path to the loaded file as Image_path
    """
    global Image_path
    try:
        # Get file path from user
        file_path = input("Enter the path to your file: ")
        
        # Validate if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Get filename and create destination path
        filename = os.path.basename(file_path)
        destination = os.path.join("uploads", filename)
        
        # Copy file to uploads folder
        shutil.copy2(file_path, destination)
        print(f"File loaded successfully to: {destination}")
        
        # Set the global Image_path variable
        Image_path = destination
        return Image_path
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Example usage:
# Image_path = get_file_from_user()
# if Image_path:
#     process_image(Image_path)
def uz():
    obj=practice()
    obj.get_file_from_user()

if __name__ == "__uz__":
    uz()