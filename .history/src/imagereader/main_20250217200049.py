import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import google.generativeai as genai
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = models.efficientnet_b0(pretrained=True)
model.eval()