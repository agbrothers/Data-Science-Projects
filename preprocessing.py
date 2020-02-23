import numpy as np
import pandas as pd
import os
import noisereduce as nr
import librosa


# Functions
def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return((arr-mn) / (mx-mn))

def processAudio(audio, label):
    noise = audio[0:25000] # ???
    audio = nr.reduce_noise(audio, noise, verbose=False)
    audio, index = librosa.effects.trim(audio, top_db=20, frame_length=512, hop_length=64)
        
    stft = np.abs(librosa.stft(audio, n_fft=512, hop_length=256, win_length=512))
    stft = np.mean(stft, axis=1)
    stft = minMaxNormalize(stft)
        
    df = pd.DataFrame([stft])
    df['Label'] = label
    return(df)




""" TRIM & REDUCE NOISE, THEN FOURIER TRANSFORM, FLATTEN, AND STORE IN CSV """

directory = r'UrbanSound/data/'
labels = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']
master_df = pd.DataFrame()

for folder in os.listdir(directory):
    if folder == '.DS_Store':
        continue
    k = 0
    for file in os.listdir(directory + '/' + folder + '/'):
        if file == '.DS_Store':
            continue
        
        audio, rate = librosa.load(directory + '/' + folder + '/' + file)
        master_df = master_df.append( processAudio(audio, folder))
        k+=1
        print(folder, k)
        
master_df.to_csv('preprocessed_audio.csv')
