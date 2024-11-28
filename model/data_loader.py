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

def extract_speech_features(audio_path, sample_rate=22050):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sample_rate, offset=0.4, duration=4)
    
    # Mel-frequency Cepstral Coefficients (MFCCs)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30) # Mean of MFCCs
    mfcc = np.mean(mfcc_features.T, axis=0) # along the row
    # print("MFCC Feat: ", mfcc_features, mfcc_features.shape)
    # print("Mean MFCC: ", mfcc, mfcc.shape)
    
    
    # Root Mean Square Energy (RMS)
    rms_features = librosa.feature.rms(y=y)
    rms = np.mean(rms_features.T, axis=0)
    # print("RMS Features: ", rms_features, rms_features.shape)
    # print("Mean RMS: ", rms, rms.shape)
    
    # Zero Crossing Rate (ZCR)
    zcr_features = librosa.feature.zero_crossing_rate(y=y)
    zcr = np.mean(zcr_features.T, axis=0)
    # print("ZCR Features: ", zcr_features, zcr_features.shape)
    # print("Mean ZCR: ", zcr, zcr.shape)
    
    # Spectral Centroid
    spectralCentroid_features = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid = np.mean(spectralCentroid_features.T, axis=0)
    # print("Spectral Centroid Features: ", spectralCentroid_features, spectralCentroid_features.shape)
    # print("Mean SPC: ", spectral_centroid, spectral_centroid.shape)
    
    # Spectral Rolloff
    spectral_rolloff_features = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    spectral_rolloff = np.mean(spectral_rolloff_features.T, axis=0)  # roll_percent instead of threshold
    # print("Spectral Rolloff Features: ", spectral_rolloff_features, spectral_rolloff_features.shape)
    # print("Mean SPCrolloff: ", spectral_rolloff, spectral_rolloff.shape)

    
    # Chroma Features
    stft = np.abs(librosa.stft(y=y))
    chroma_features = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    chroma_stft = np.mean(chroma_features.T, axis=0)
    # print("Chroma Features: ", chroma_features, chroma_features.shape)
    # print("Chroma STFT: ", chroma_stft, chroma_stft.shape)

    
    # Combine all features into a single feature vector
    features = np.hstack((mfcc, rms, zcr, spectral_centroid, spectral_rolloff, chroma_stft))
    # print(features.shape)
    
    return features







                           # "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/model/data.csv"



def getDatapoints(csv_path = "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/model/data.csv", batch_size = 64):
    data = pd.read_csv(csv_path)

    X, Y = [], []
    for index, row in data.iterrows():
        # print(index, data["audio"][index], data["emotion"][index], data["class"][index])
        audioPath = data["audio"][index]
        audioClass = data["class"][index]
        # vector = getMelVector(audio_path=audioPath, targetDuration = 4.0)
        vector = extract_speech_features(audio_path=audioPath)
        X.append(vector)
        Y.append(audioClass)
        # break
        

        if(index% 1000 == 0):
            print(f"{index} data points are processed")
        
        # showMelPlot(vector)
        # print(vector)
        # if(index==5): #for Testing
        #     break
    
    X = np.array(X)
    # X = X.T
    # X = np.expand_dims(X, -1)
    Y = np.array(Y)
    Y = Y.astype(np.int64)
    # print(X)
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
    

    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    Y_tensor = torch.tensor(Y, dtype=torch.long)  # Use torch.long for classification
    # print(X_tensor.shape, Y_tensor.shape)

    dataset = TensorDataset(X_tensor, Y_tensor)

    trainingSize = int(len(dataset) * 0.9)
    testSize = len(dataset) - trainingSize 
    train_dataset, test_dataset = random_split(dataset, [trainingSize, testSize])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train Data Size: {len(train_dataset)} and Test Data Size: {len(test_dataset)}")
    
    return train_loader, test_loader

    
# trainLoader, testLoader = getDatapoints()
# print(len(trainLoader.dataset), len(testLoader.dataset))

# print(data.head())

