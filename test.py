import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from model import Classifier
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier()
model.to(device)

# Load saved parameters
model.load_state_dict(torch.load("tomjerry.pth", weights_only=True))
model.eval()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 1e-4
)

# Define the same transformation as when training
transform = transforms.Compose([transforms.Resize((227, 227)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])])


class_names = ['jerry', 'tom', 'none', 'both']

# Change image file name to make inference
img = Image.open("test_img/jerry3.jpg").convert("RGB")
img = transform(img)
img = img.unsqueeze(0)
img = img.to(device)

# Code to predict on single image
with torch.inference_mode():
    res = model(img)
    prob = F.softmax(res, dim=1)
    preds = prob.argmax(dim=1)
    print(class_names[preds.item()])
    print(prob)



