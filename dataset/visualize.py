import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import librosa

import pandas as pd

def plotCountPlot(dataset_path, save_dir, filename):
    base_dir = dataset_path

    # Get the list of subdirectories (e.g., 'sad', 'happy', etc.)
    sub_dirs = [sub_dir for sub_dir in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, sub_dir))]

    # Prepare a list of tuples (sub_dir, number_of_files_in_sub_dir)
    sub_dir_counts = []

    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_dir, sub_dir)
        file_count = len([f for f in os.listdir(sub_dir_path) if os.path.isfile(os.path.join(sub_dir_path, f))])
        sub_dir_counts.append((sub_dir, file_count))

    # Convert to a DataFrame for easier plotting with seaborn
    df = pd.DataFrame(sub_dir_counts, columns=['Subdirectory', 'File Count'])

    # Plotting the countplot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Subdirectory', y='File Count', data=df, hue='Subdirectory', palette='viridis', legend=False)

    # Annotating the bars with their respective file counts
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha='center', va='center', 
        fontsize=12, color='black', 
        xytext=(0, 5), textcoords='offset points')

    # Customize the plot
    plt.title('Count of Each Audio ', loc ='center', fontsize=16)
    plt.xlabel('Emotions', fontsize=14)
    plt.ylabel('Number of Audio Files', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_filename = filename
    plt.savefig(save_dir + output_filename)
    plt.show()



def wavePlot(datapath, WaveformName, save_dir, filename):
# Load an audio file
    y, sr = librosa.load(datapath, sr=None)

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(y)
    plt.title(WaveformName)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.savefig(save_dir + filename)
    plt.show()


def spectogramPlot(datapath, WaveformName, save_dir, filename):

# Compute the spectrogram
    y, sr = librosa.load(datapath, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(WaveformName)
    plt.savefig(save_dir + filename)
    plt.show()


if __name__=="__main__":
    
    # plotCountPlot("/Users/ishananand/Desktop/ser/combined_dataset", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/", "initial_countplot.png")
    # wavePlot("/Users/ishananand/Desktop/ser/combined_dataset/angry/03-01-05-01-01-01-01.wav", "Anger WaveForm", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/waveform/", "angerWaveform.png")
    # wavePlot("/Users/ishananand/Desktop/ser/combined_dataset/bored/03a04Lc.wav", "Bored WaveForm", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/waveform/", "boredWaveform.png")
    # wavePlot("/Users/ishananand/Desktop/ser/combined_dataset/disgust/03-01-07-01-01-01-01.wav", "Disgust WaveForm", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/waveform/", "disgustWaveform.png")
    # wavePlot("/Users/ishananand/Desktop/ser/combined_dataset/fear/03-01-06-01-01-01-01.wav", "Fear WaveForm", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/waveform/", "fearWaveform.png")
    # wavePlot("/Users/ishananand/Desktop/ser/combined_dataset/happy/03-01-03-01-01-01-01.wav", "Happy WaveForm", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/waveform/", "happyWaveform.png")
    # wavePlot("/Users/ishananand/Desktop/ser/combined_dataset/neutral/03-01-01-01-01-01-01.wav", "Neutral WaveForm", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/waveform/", "neutralWaveform.png")
    # wavePlot("/Users/ishananand/Desktop/ser/combined_dataset/sad/03-01-04-01-01-01-01.wav", "Sad WaveForm", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/waveform/", "sadWaveform.png")
    # wavePlot("/Users/ishananand/Desktop/ser/combined_dataset/surprise/03-01-08-01-01-01-01.wav", "Surprise WaveForm", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/waveform/", "surpriseWaveform.png")

    spectogramPlot("/Users/ishananand/Desktop/ser/combined_dataset/angry/03-01-05-01-01-01-01.wav", "Anger Spectogram", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/spectogram/", "angerSpectogram.png")
    spectogramPlot("/Users/ishananand/Desktop/ser/combined_dataset/bored/03a04Lc.wav", "Bored Spectogram", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/spectogram/", "boredSpectogram.png")
    spectogramPlot("/Users/ishananand/Desktop/ser/combined_dataset/disgust/03-01-07-01-01-01-01.wav", "Disgust Spectogram", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/spectogram/", "disgustSpectogram.png")
    spectogramPlot("/Users/ishananand/Desktop/ser/combined_dataset/fear/03-01-06-01-01-01-01.wav", "Fear Spectogram", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/spectogram/", "fearSpectogram.png")
    spectogramPlot("/Users/ishananand/Desktop/ser/combined_dataset/happy/03-01-03-01-01-01-01.wav", "Happy Spectogram", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/spectogram/", "happySpectogram.png")
    spectogramPlot("/Users/ishananand/Desktop/ser/combined_dataset/neutral/03-01-01-01-01-01-01.wav", "Neutral Spectogram", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/spectogram/", "neutralSpectogram.png")
    spectogramPlot("/Users/ishananand/Desktop/ser/combined_dataset/sad/03-01-04-01-01-01-01.wav", "Sad Spectogram", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/spectogram/", "sadSpectogram.png")
    spectogramPlot("/Users/ishananand/Desktop/ser/combined_dataset/surprise/03-01-08-01-01-01-01.wav", "Surprise Spectogram", "/Users/ishananand/Desktop/ser/Speech-Emotion-Recognition/images/spectogram/", "surpriseSpectogram.png")