import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Classifier(nn.Module):
    def __init__(self, num_classes = 4):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=11, stride=4),
            nn.ReLU(inplace = True),

            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(64, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=2),

            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)

        )

    def forward(self, x):
        return self.model(x)
    

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))