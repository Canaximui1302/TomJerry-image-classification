import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import Classifier

transform = transforms.Compose([transforms.Resize((227, 227)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])])

DATA_DIR = "data/tom_jerry_dataset/tom_and_jerry"
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Define train dataset and testdataset
# Then split them from the original full dataset
# Here batchsize is 32

train_size = int(0.8 * len(dataset))
test_size = int(len(dataset) - train_size)

train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)


# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier(num_classes=4)
model.to(device)

# Use Cross Entropy as loss function
loss_func = nn.CrossEntropyLoss()

# Define optimizer, using Adam for fast convergence
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 1e-4
)

# Number of training epochs
num_epochs = 15

# Arrays to store metrics for visualization
training_losst_array = []
training_accuracy_array = []

# Traning loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0.
    total = 0.
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward
        outputs = model(inputs)
        loss = loss_func(outputs, labels)  # Calculate loss

        # Backward
        optimizer.zero_grad()  # Zero out previous gradients
        loss.backward()  # Backpropagate to calculate gradients
        optimizer.step()  # Update weights
        
        running_loss += loss.item()

        # Accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Calculate accuracy and loss to print after each epoch
    # Helps to know when to stop early in case of overfitting
    val_acc = 100 * val_correct / val_total
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    training_losst_array.append(train_loss)
    training_accuracy_array.append(train_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")



# Save the model's trained parameters 
torch.save(model.state_dict(), "tomjerry.pth")
print("Model saved!")

# Save the metrics for visualization
np.savez("metrics.npz", train_losses=training_losst_array, train_accuracies=training_accuracy_array)
print("Metrics saved!")