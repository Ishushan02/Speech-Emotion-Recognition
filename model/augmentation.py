import numpy as np
import IPython.display as ipd
import librosa
import soundfile as sf
import noisereduce as nr
import os


'''

There are many sound augmentation techniques which we can use, like adding noise, stretching, pitching, 
shifting the noise, now as this Project completely deals with the Emotion recognition technique of the 
dataset increasing the pitch, shiftness and stretching seems like the dataset will be coagulated, 
which means that the anger voices would be highly pitched whereas the sad voices would be lowly pitched so 
meddling with the pitches, shiftness and stretching property of Audio's might alter the audio characterstic
entirely. So, for that case I am using only Noise addition to the dataset which will increase our dataset
size and also noise addition seems to be realistic as the normal audio's will be with noise.

The Longest Audio Length in our dataset is of 7 Seconds.

'''


def noise(data):
    """
    Adds Gaussian noise to the input audio data. This will just add a normal distr Gaussian Noise with factor of 0.032
    """
    noise_amp = 0.032 * np.random.uniform() * np.amax(data)
    noise_data = data + noise_amp * np.random.normal(size=data.shape[0])
    return noise_data

def add_noise_to_audio(input_path, output_path):
    """
    Reads audio from input_path, adds noise, and saves it to output_path.
    """
    # Load the audio file
    data, sr = librosa.load(input_path, sr=None)  # sr=None preserves the original sample rate
    
    # Apply noise to the audio
    noisy_data = noise(data)
    
    # Save the noisy audio to the output file
    sf.write(output_path, noisy_data, sr)

if __name__=="__main__":

    # audio_file_path = "/Users/ishananand/Desktop/ser/combined_dataset/angry/1007_IWW_ANG_XX.wav"
    # output = "/Users/ishananand/Desktop/ser/combined_dataset/angry/1007_IWW_ANG_XX_noised.wav"

    # add_noise_to_audio(audio_file_path, output)

    # Doing the above procedure for all of the Dataset

    root_path = "/Users/ishananand/Desktop/ser/combined_dataset"
    allaudios = os.listdir(root_path)

    for i in range(len(allaudios)):
        audio_path = root_path + "/" + allaudios[i]
        each_audio_path = os.listdir(audio_path)
        max_audio = 0
        for j in range(len(each_audio_path)):
            current_audio_path = audio_path + "/" + each_audio_path[j]
            noised_audio_path = audio_path + "/noised_" + each_audio_path[j]
            # add_noise_to_audio(current_audio_path, noised_audio_path)
            # data, sr = librosa.load(current_audio_path)
            # max_audio = max(max_audio, len(data/sr))
            # print("Name of data: ", current_audio_path, "Shape of data: ", data.shape, " and Duration Of Audio: ", len(data/sr))
        print(f"All Audio's of {allaudios[i]} is been Augmented")
    # print(max_audio)
    print("All Audios are Augmented by adding noise to them")
