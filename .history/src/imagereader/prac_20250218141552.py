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

from dotenv import load_dotenv

load_dotenv()
#image_path = './images/thyaemc.jpeg'
#image_path=os.path(Image_path)
class practice:
    def __init__(self):
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.eval()
        self.image_path = None  # Add this to store the path
       

    def get_file_from_user(self):
        """
        Get file path from user input and load the file
        Returns:
            str: Path to the loaded file as Image_path
        """
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
            print(f"file path is {destination}")
            # Set the global Image_path variable
            self.image_path = destination
            file_paths = [self.image_path]
            titles = [filename]
            return file_paths, titles
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def preprocess_image(self, image_path=None):
        if image_path is None:
            image_path = self.image_path
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)

    # Function to Extract Features
    def extract_features(self, image_tensor=None):
        """Extract features from image tensor"""
        if image_tensor is None:
            # Get tensor from preprocessed image if not provided
            image_tensor = self.preprocess_image()
            print(f"image tensor{image_tensor}")
        with torch.no_grad():
            features = self.model(image_tensor)  # Extract features
        return features.numpy().flatten()

    # Function to create image embeddings
    def create_image_embedding(self, image_path=None):
        if image_path is None:
            image_path = self.image_path
        # Load the pretrained model
        # model = InceptionResnetV1(pretrained='vggface2').eval()
        # model = models.efficientnet_b0(pretrained=True)
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.eval()    
        try:
            input_tensor = self.preprocess_image(image_path)
            print(f"image embedding{input_tensor}")
            with torch.no_grad():
                embeddings = self.model(input_tensor)  # embedding important line
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
def uz():
    obj = practice()
    obj.get_file_from_user()
    if obj.image_path:
        # Now these will work in sequence
        tensor = obj.preprocess_image()
        features = obj.extract_features(tensor)
        embeddings = obj.create_image_embedding()

if __name__ == "__uz__":
    uz()