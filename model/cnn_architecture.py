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
import time
import os
import data_loader



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
        out = self.fc6(x)
        # print(x.shape)
        # out = torch.argmax(x, dim=1).long()
        # print(out)

        return out


class CNNArchitecture1D(nn.Module):

    def __init__(self, classes):
        super(CNNArchitecture1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.Pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        
        self.fc1 = nn.Linear(768 , 128)  # Adjusted for the output size after pooling
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 256)
        # self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, classes)

    def forward(self, x):

        # Convolutional layers with ReLU activations and pooling
        x = Fn.relu(self.conv1(x))
        x = self.Pool(x)
        # x = Fn.relu(self.conv2(x))
        # x = self.Pool(x)
        # x = Fn.relu(self.conv3(x))
        # x = self.Pool(x)
        x = Fn.relu(self.conv4(x))
        x = self.Pool(x)

        x = x.view(x.size(0), -1)  # Flatten the output to (batch_size, num_features)
        print(x.shape)

        # Fully connected layers with ReLU activations
        x = Fn.relu(self.fc1(x))
        # x = Fn.relu(self.fc2(x))
        # x = Fn.relu(self.fc3(x))
        # x = Fn.relu(self.fc4(x))
        x = Fn.relu(self.fc5(x))

        # Final output layer (no ReLU)
        out = self.fc6(x)
        # print(x.shape)
        # out = torch.argmax(x, dim=1).long()
        # print(out)

        return out




# Testing if the Architecture is working or not

# model = CNNArchitecture1D(6)
# img = torch.randn(64, 1, 40)
# tensor = torch.tensor(img)
# print(model(img).shape)

trainDataLoader, testDataLoader = data_loader.getDatapoints()



device = torch.device("cpu")
epochs = 500
model = CNNArchitecture1D(classes=6)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
lossFn = nn.CrossEntropyLoss()


def train(trainDataLoader, epochs, modelPath):
    print(f"Training Started for  this {epochs} rpochs")
    for param in model.parameters():
        param.requires_grad = True 
        lossFn = nn.CrossEntropyLoss()

    for each_epoch in range(epochs):
            epoch_loss = 0
            correct_predictions = 0 
            total_samples = 0

            for batch_id, (trainX, trainY) in enumerate(trainDataLoader):
                
                start_time = time.time()

                # Add channel dimension (for Conv2d)
                trainX = trainX.unsqueeze(1)  # Adds a channel dimension at position 1
                trainX = trainX.to(device)
                trainY = trainY.long()
                trainY = trainY.to(device)
                print(trainX.shape)
                # Forward pass
                pred = model(trainX)
                
                # print("Predictions dtype:", pred.dtype)  # This should be torch.float32
                # trainY.dtype = torch.long()
                # print("Targets dtype:", trainY.dtype) 
                # Compute the loss
                lossval = lossFn(pred, trainY)
                
                optimizer.zero_grad()
                
                # Check if the loss tensor requires gradients
                # print("Check: ", lossval.requires_grad)  # This should print True now

                # Backward pass
                lossval.backward()

                # Update model parameters
                optimizer.step()
                
                epoch_loss += lossval.item()  # Add batch loss to epoch loss
                
                end_time = time.time()
                batch_time = end_time - start_time  # Time taken for the batch
                # print(f"Batch {batch_id + 1}, Time per batch: {batch_time:.4f} seconds")

                # break  # For debugging, you can remove this to train on the full dataset
                with torch.no_grad():  # No gradient computation for accuracy
                    predictions = torch.argmax(pred, dim=1)  # Get predicted class labels
                    correct_predictions += (predictions == trainY).sum().item()  # Count correct predictions
                    total_samples += trainY.size(0)  # Update total number of samples
            average_loss = epoch_loss / len(trainDataLoader)  # Average loss
            accuracy = correct_predictions / total_samples * 100  # Accuracy as percentage

            # Display metrics for the epoch
            print(f"Epoch {each_epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}%")
            if(accuracy >= 99):
                print(f"Stopping the Training as Accuracy has reached to Maximum")
                break
        
    
    print(f"Training Completed for  this {epochs}")
    torch.save(model.state_dict(), f = modelPath)
    print(f"Model saved to Model Path")



def test(testDataLoader, modelPath):
    test_model = CNNArchitecture1D(classes=6)
    # Load the saved model weights
    test_model.load_state_dict(torch.load(modelPath))

    # Move the model to the appropriate device (CPU or GPU)
    # 5. Set the model to evaluation mode (important for inference)

    # the model to evaluation mode (important for inference)
    test_model.eval()

    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in testDataLoader:

            x = x.unsqueeze(1)  # Adds a channel dimension at position 1
            y = y.long()
            x = x.to(device)
            y = y.to(device)
            
            scores = test_model(x)

            predictions = torch.argmax(scores, dim=1)
            # print(predictions, "-----", y)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    accuracy = num_correct / num_samples * 100  # Accuracy as percentage
    
    print(f"Test Accuracy is  {accuracy}")




def testnewAudio(modelPath, testAudioPath):

    test_model = CNNArchitecture1D(classes=6)
    # Load the saved model weights
    test_model.load_state_dict(torch.load(modelPath))
    test_model.eval()

    allAudios = os.listdir(testAudioPath)
    emotion_class = {
        0: "happy",
        1: "angry",
        2: "fear",
        3: "sad",
        4: "disgust",
        5: "neutral"
    }
    X, Y = [], []
    for each_audio in allAudios:
        # print(rootpath + "/" +each_audio)

        # test_audio = getMelVector(rootpath + "/" +each_audio, 4)
        test_audio = data_loader.extract_speech_features(audio_path=testAudioPath + "/" +each_audio)
        # print(test_audio)
        
        X.append(test_audio)
        if("angry" in each_audio):
            Y.append(1)
        elif("disgust" in each_audio):
            Y.append(4)
        elif("happy" in each_audio):
            Y.append(0)
        elif("sad" in each_audio):
            Y.append(3)
        elif("neutral" in each_audio):
            Y.append(5)
        elif("fear" in each_audio):
            Y.append(2)
        

    X = np.array(X)
    Y = np.array(Y)
    Y = Y.astype(np.int64)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.long)  # Use torch.float32 for regression, torch.long for classification

    dataset = TensorDataset(X_tensor, Y_tensor)
    customLoader = DataLoader(dataset, batch_size=1, shuffle=False)
    # customLoader.dataset
    with torch.no_grad():
        for x, y in customLoader:

            x = x.unsqueeze(1)  # Adds a channel dimension at position 1
            x = x.to(device)
            y = y.to(device)
            print(x.shape)
            scores = test_model(x)
            print(scores)
            y = int(y.to(torch.int32))
            predictions = torch.argmax(scores, dim=1)
            # tensor_int = int(predictions.to(torch.int32))
            # print(customLoader.)
            print(predictions, "-----", y)
            # print(f" The true Value is {emotion_class[y]} and predicted class is {emotion_class[tensor_int]}")
            # print(scores)

            
            # print(predictions)

        test_model.train()



modelPath = "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/model_weight/CNNModel_Feat2.pth"
train(trainDataLoader, 20, modelPath)
test(testDataLoader, modelPath)

newAudiosPath = "/Users/ishananand/Desktop/ser/testAudios"
testnewAudio(modelPath, newAudiosPath)

