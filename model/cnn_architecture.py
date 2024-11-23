import torch
from torch import nn
from torch.nn import functional as Fn
from torch import optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import getDatapoints
import time


class CNNArchitecture(nn.Module):

    def __init__(self, classes):
        super(CNNArchitecture, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.Pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(256 , 1024)  # Adjusted for the output size after pooling
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, classes)

    def forward(self, x):

        # Convolutional layers with ReLU activations and pooling
        x = Fn.relu(self.conv1(x))
        x = self.Pool(x)
        x = Fn.relu(self.conv2(x))
        x = self.Pool(x)
        x = Fn.relu(self.conv3(x))
        x = self.Pool(x)
        x = Fn.relu(self.conv4(x))
        x = self.Pool(x)

        x = x.view(x.size(0), -1)  # Flatten the output to (batch_size, num_features)
        # print(x.shape)

        # Fully connected layers with ReLU activations
        x = Fn.relu(self.fc1(x))
        x = Fn.relu(self.fc2(x))
        x = Fn.relu(self.fc3(x))
        x = Fn.relu(self.fc4(x))
        x = Fn.relu(self.fc5(x))

        # Final output layer (no ReLU)
        x = self.fc6(x)
        # print(x.shape)
        out = torch.argmax(x, dim=1).float()
        # print(out)

        return out


# model = CNNArchitecture(7)
# img = torch.randn(64, 1, 128, 173)
# tensor = torch.tensor(img)
# print(model(img).shape)


trainDataLoader, testDataLoader = getDatapoints()



device = torch.device("cpu")
epochs = 500
model = CNNArchitecture(classes=6)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
lossFn = nn.CrossEntropyLoss()

# model.load_state_dict(torch.load('/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/model_weight/CNNModel.pth'))


for each_epoch in range(1):
    epoch_loss = 0
    correct_predictions = 0 
    total_samples = 0

    for batch_id, (trainX, trainY) in enumerate(trainDataLoader):
        
        start_time = time.time()

        trainX = trainX.unsqueeze(1)  # Adds a channel dimension at position 1
        trainX = trainX.to(device)
        trainY = trainY.to(device)
        trainX = trainX.requires_grad_(True)

        print(trainX.shape, trainY.shape)

        model.train()
        # forward
        pred = model(trainX)

        print(trainY, pred)
        # pred = torch.float32(pred)


        # print(pred.shape, trainY.shape)
        # predy = torch.argmax(pred, dim=1)
        # break

        # backward
        
        lossval = lossFn(pred, trainY)
        optimizer.zero_grad()
        

        # grad descent
        lossval.backward()

        optimizer.step()
        epoch_loss += lossval.item()  # Add batch loss to epoch loss
        end_time = time.time()
        batch_time = end_time - start_time  # Time taken for the batch
        print(f"Batch {batch_id + 1}, Time per batch: {batch_time:.4f} seconds")

        break
        with torch.no_grad():  # No gradient computation for accuracy
            # predictions = torch.argmax(pred, dim=1)  # Get predicted class labels
            correct_predictions += (predy == trainY).sum().item()  # Count correct predictions
            total_samples += trainY.size(0)  # Update total number of samples
    average_loss = epoch_loss / len(trainDataLoader)  # Average loss
    accuracy = correct_predictions / total_samples * 100  # Accuracy as percentage

    # Display metrics for the epoch
    print(f"Epoch {each_epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}%")




torch.save(model.state_dict(), f = "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/model_weight/CNNModel.pt")
print(f"Model saved to Model Path")






