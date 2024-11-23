import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
I am not Choosing the normal MelSPectogram because if you see it's plot the most part's in it
are dark and doesn't have much datapoints feature hence the model will tend to learn mostly the
darker part of the spectogram, whereas when you see the logartithmic melSpectogram(representation
of what human hears, the features can be learned from it).. so inculcating the Logarithmic
MelSpectogram
'''

def showMelPlot(logMelVector):
    librosa.display.specshow(data= logMelVector, x_axis="time", y_axis="mel", sr = 22050)
    plt.colorbar()
    plt.show()


def getMelVector(audio_path, targetDuration = 5.0):
    audio_series, sampling_rate = librosa.load(audio_path, sr=22050, mono=True)

    target_samples = int(targetDuration * sampling_rate) # Total Number of Samples from the Audio 

    if len(audio_series) < target_samples:
        padding = target_samples - len(audio_series)
        audio_series = np.pad(audio_series, (0, padding), mode='constant') # padding with 0's if less than standard length
    else:
        audio_series = audio_series[:target_samples]  # trimming the audio length if it's more than required length

    mel_spectogram = librosa.feature.melspectrogram(y=audio_series, sr = 22050, n_fft=2048, hop_length=512)
    logarithmic_melSpectogram = librosa.power_to_db(mel_spectogram, ref=np.max)

    # melSpectogram = librosa.power_to_db(mel_spectogram) # normal Melspectogram where datapoint's are darker mostly
    # print(logarithmic_melSpectogram.shape, audio_path)
    

    return logarithmic_melSpectogram



def getDatapoints(csv_path = "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/model/data.csv"):
    data = pd.read_csv(csv_path)

    X, Y = [], []
    for index, row in data.iterrows():
        # print(index, data["audio"][index], data["emotion"][index], data["class"][index])
        audioPath = data["audio"][index]
        audioClass = data["class"][index]
        vector = getMelVector(audio_path=audioPath, targetDuration = 4.0)
        X.append(vector)
        Y.append(audioClass)
        
        
        # showMelPlot(vector)
        # print(vector)
        # break
    
    X = np.array(X)
    Y = np.array(Y)
    Y = Y.astype(np.int64)
    print(f"Completed collecting all the data points X: {X.shape} and Y: {Y.shape}")


    # Calculate mean and std for normalization (on the training set)
    # mean = np.mean(X, axis=(0, 1, 2), keepdims=True)  # Compute mean over height, width, and channels
    # std = np.std(X, axis=(0, 1, 2), keepdims=True)  # Compute std over height, width, and channels

    # max = np.max(X, axis=(0, 1, 2), keepdims=True)  
    # min = np.min(X, axis=(0, 1, 2), keepdims=True)  
    # print(max, min)

    # Normalize the images with the calculated mean and std
    # X = (X - mean) / std  # This normalization will use the same mean and std for all images
    # print(f" Mean of DataPoints: {mean} and STD of DataPoints: {std}")
    

    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y, dtype=torch.long)  # Use torch.float32 for regression, torch.long for classification

    dataset = TensorDataset(X_tensor, Y_tensor)

    trainingSize = int(len(dataset) * 0.9)
    testSize = len(dataset) - trainingSize 
    train_dataset, test_dataset = random_split(dataset, [trainingSize, testSize])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

    
trainLoader, testLoader = getDatapoints()
print(len(trainLoader.dataset), len(testLoader.dataset))

# print(data.head())

