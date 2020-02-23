import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import librosa
import noisereduce as nr


""" Lots of plotting to help visualize & improve the quality of our input """

def plot_noise_reduction(file):
    # Plots Noise Reduction Visualizations for the provided audio files
    if file == '.DS_Store':
        return()
    
    # Split the audio clip into 8 even intervals
    audio, sr = librosa.load(directory + label + file)
    step_size = int(audio.shape[0]/8)
    intervals = [ [i, i + step_size] for i in range(0, audio.shape[0] - step_size, step_size)]
    
    # Find the variance of the amplitude over each interval and take the min as the noisy interval
    std = [np.std(audio[i[0]:i[1]]) for i in intervals]
    noise_interval = intervals[np.argmin(std)]
    noise = audio[noise_interval[0]:noise_interval[1]]
    noise_plot = np.zeros(len(audio))
    noise_plot[noise_interval[0]:noise_interval[1]] = audio[noise_interval[0]:noise_interval[1]]
    reduced_noise = nr.reduce_noise(audio, noise, verbose=False)

    fig = plt.figure(figsize=(9,9))
    ax1 = fig.add_subplot(3,1,1)
    plt.title(label.replace('/',''))
    ax1.tick_params(labelsize=6)
    ax1 = plt.plot(audio, color='k', lw=0.5, alpha=0.9)
    
    ax2 = fig.add_subplot(3,1,2)
    plt.title('Noise Highlighted')
    ax2.tick_params(labelsize=6)
    ax2 = plt.plot(audio, color='k', lw=0.5, alpha=1)
    ax2 = plt.plot(noise_plot, color='r', lw=0.5, alpha=0.9)
        
    ax3 = fig.add_subplot(3,1,3)
    plt.title('Noise Reduced')
    ax3.tick_params(labelsize=6)
    ax3 = plt.plot(audio, color='b', lw=0.5, alpha=0.3)
    ax3 = plt.plot(reduced_noise, color='k', lw=0.5, alpha=0.9)

    plt.tight_layout()


""" SAMPLE RANDOM AUDIO FILES """
directory = r'UrbanSound/data/'
folders = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']

# Plot the noise reduction of three random audio clips from the directory
num_samples = 3
k = 0
while k < num_samples:
    label = random.choice(folders) + '/'
    index = random.randint(0, len(os.listdir(directory + label))-1)
    file = os.listdir(directory + label)[index]
    
    plot_noise_reduction(file)
    k+=1
