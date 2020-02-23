import numpy as np
import pandas as pd
import os
import noisereduce as nr
import librosa

"""
Preprocessing audio files from the UrbanSounds dataset for use in training a Neural Network
in audio classification problems.  Each audio file is converted to a vector with average
amplitude per frequency bin via Short Time Fourier Transform & mean flattening.  Audio 
vectors are stored in a dataframe and exported for future use.  
"""

# Functions
def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return((arr-mn) / (mx-mn))

def processAudio(audio, label):
    noise = audio[0:25000]
    audio = nr.reduce_noise(audio, noise, verbose=False)
    audio, index = librosa.effects.trim(audio, top_db=20, frame_length=512, hop_length=64)
        
    stft = np.abs(librosa.stft(audio, n_fft=512, hop_length=256, win_length=512))
    stft = np.mean(stft, axis=1)
    stft = minMaxNormalize(stft)
        
    df = pd.DataFrame([stft])
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
        audio, rate = librosa.load(directory + '/' + folder + '/' + file)
        master_df = master_df.append( processAudio(audio, folder))
        
master_df.to_csv('preprocessed_audio.csv')
