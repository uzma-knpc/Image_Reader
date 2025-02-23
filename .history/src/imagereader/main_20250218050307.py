import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import google.generativeai as genai
import os
from facenet_pytorch import MTCNN, InceptionResnetV1

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
import shutil
import requests
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import datetime
from PIL import Image
from transformers import pipeline


from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#Load the pretrained model
model = models.efficientnet_b0(pretrained=True)
model.eval()

# Preprocessing function to transform the image into a tensor
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])
    return transform(image).unsqueeze(0)

# Function to Extract Features
def extract_features(image_tensor):
    with torch.no_grad():
        features = model(image_tensor)  # Extract features
    return features.numpy().flatten()

# Function to create image embeddings
def create_image_embedding(image_path):
    try:
        input_tensor = preprocess_image(image_path)
        with torch.no_grad():
            embeddings = model(input_tensor) # ebedding important line
        return embeddings.squeeze().numpy()
    except Exception as e:
        print("Error:", e)
        return None

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_inception_resnet_v1.pth")
def save_image_from_url(image_url, image_name):
    """
    Downloads an image from a URL or copies it from a local file path,
    and saves it to the 'images' folder.
    """
    try:
        # Create 'images' folder if it doesn't exist
        if not os.path.exists("images"):
            os.makedirs("images")

        image_path = os.path.join("images", image_name)

        # Check if the input is a URL or a local file path
        if image_url.startswith("http://") or image_url.startswith("https://"):
            # Download the image from a URL
            response = requests.get(image_url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Save the image in chunks
            with open(image_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Image downloaded and saved to: {image_path}")

            print(f"File exists: {os.path.exists(image_url)}")
            print(f"Is a file: {os.path.isfile(image_url)}")
            print(f"Invalid image URL or file path: {image_url}")


        elif os.path.exists(image_url) and os.path.isfile(image_url):
            # Debugging: Print file path validation info
            print(f"Local file path is valid: {image_url}")

            # Copy the image from a local file path
            shutil.copy(image_url, image_path)
            print(f"Image copied from local path and saved to: {image_path}")

        else:
            # Debugging: Print error for invalid paths
            print(f"file{image_path}")
            print(f"File exists: {os.path.exists(image_url)}")
            print(f"Is a file: {os.path.isfile(image_url)}")
            print(f"Invalid image URL or file path: {image_url}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

#save image into current director
save_image_from_url(Image_path, file_name)
m = create_image_embedding(Image_path)
file_paths = [
  
    Image_path,
   # "images/Nlung.jpg" ,  # Replace with actual file paths for each image

]
titles = [
   file_name,
   
]
# Function to load an image and convert it to grayscale
def load_image(file_path):
    img = Image.open(file_path).convert("L")
    img_array = np.array(img)
    return img_array

# Function to normalize the image intensity
def normalize_image(img):
    img_min, img_max = img.min(), img.max()
    normalized_img = (img - img_min) / (img_max - img_min)
    return normalized_img
# Diagnostic function with criteria
def diagnose_image(img):
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    threshold_mean = 0.5
    threshold_std = 0.1

    # Criteria for diagnosis
    if mean_intensity > threshold_mean and std_intensity > threshold_std:
        return "Abnormal scan detected"
    else:
        return "Scan appears normal"
    
    # Function to calculate image metrics
def calculate_metrics(img):
    return {
        "Mean Intensity": np.mean(img),
        "Standard Deviation": np.std(img),
        "Minimum Intensity": np.min(img),
        "Maximum Intensity": np.max(img)
    }