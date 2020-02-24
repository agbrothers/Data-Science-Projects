import numpy as np
import pandas as pd
import os
import noisereduce as nr
import librosa

"""
Preprocessing audio files from the UrbanSounds dataset for use in training a Neural Network
in sound classification problems.  Each audio file is converted to a vector with entries 
corresponding to average amplitude per frequency bin via Short Time Fourier Transform and
flattening by mean.  Audio vectors are stored in a dataframe and exported for future use.  
"""

# Functions
def center(arr):
    mn = np.min(arr)
    return(arr-mn)

def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return((arr-mn) / (mx-mn))

def findNoise(audio):
    # Split the audio clip into 8 even intervals
    step_size = int(audio.shape[0]/8)
    intervals = [[i, i + step_size] for i in range(0, audio.shape[0] - step_size, step_size)]
    
    # Find the std of the centered amplitudes over each interval and take the interval with smallest std as the noise
    std = [np.std(center(audio[i[0]:i[1]])) for i in intervals]
    noisy_part = intervals[np.argmin(std)]
    noise = audio[noisy_part[0]:noisy_part[1]]
    return(noise)

def processAudio(file, label):
    audio,sr = librosa.load(file)
    audio = nr.reduce_noise(audio, findNoise(audio), verbose=False)
    audio,_ = librosa.effects.trim(audio, top_db=20, frame_length=512, hop_length=64)

    n_mels = 257
    mel = librosa.feature.melspectrogram(audio, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = np.mean(mel, axis=1)
    mel = minMaxNormalize(mel)
    df = pd.DataFrame([mel])
    
    df['Label'] = label
    return(df)


""" MAIN """
directory = r'UrbanSound/data/'
labels = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']
master_df = pd.DataFrame()

for folder in os.listdir(directory):
    if folder == '.DS_Store':
        continue
    for file in os.listdir(directory + '/' + folder + '/'):
        if file == '.DS_Store':
            continue  
        path = directory + '/' + folder + '/' + file
        master_df = master_df.append(processAudio(path, folder))
        
master_df.to_csv('audio_features.csv')
