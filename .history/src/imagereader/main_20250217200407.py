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
