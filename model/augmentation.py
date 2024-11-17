import numpy as np
import IPython.display as ipd
import librosa
import soundfile as sf
import noisereduce as nr


'''

There are many sound augmentation techniques which we can use, like adding noise, stretching, pitching, 
shifting the noise, now as this Project completely deals with the Emotion recognition technique of the 
dataset increasing the pitch, shiftness and stretching seems like the dataset will be coagulated, 
which means that the anger voices would be highly pitched whereas the sad voices would be lowly pitched so 
meddling with the pitches, shiftness and stretching property of Audio's might alter the audio characterstic
entirely. So, for that case I am using only Noise addition to the dataset which will increase our dataset
size and also noise addition seems to be realistic as the normal audio's will be with noise.

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

    audio_file_path = "/Users/ishananand/Desktop/ser/combined_dataset/angry/1007_IWW_ANG_XX.wav"
    output = "/Users/ishananand/Desktop/ser/combined_dataset/angry/1007_IWW_ANG_XX_noised.wav"

    add_noise_to_audio(audio_file_path, output)

    
    # noised_data = noise(audio_file)
    print("ASBCSDLJK")