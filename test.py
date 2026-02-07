import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import Classifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier()
model.to(device)

model.load_state_dict(torch.load("tomjerry.pth", weights_only=True))
model.eval()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 1e-4
)

transform = transforms.Compose([transforms.Resize((227, 227)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])])


class_names = ['jerry', 'tom', 'none', 'both']  # adjust if needed

