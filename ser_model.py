import matplotlib.pyplot as plt                       # Allows you to plot things
import librosa                                        # Python package for music and audio analysis
import librosa.display                                # Allows you to display audio files 
import os                                             # The OS module in Python provides a way of using operating system-dependent functionality.
import scipy.io.wavfile                               # Open a WAV files
import numpy as np                                    # Used for working with arrays
import fastai
# import glob                                         # Used to return all file paths that match a specific pattern

from fastai import *                                 
from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders
from fastai.tabular.all import *
from fastai.text.all import *
from fastai.vision.widgets import *

import struct                                         # Unpack audio data into integers
import time
import sounddevice
from tkinter import TclError
from scipy.fftpack import fft                         # Imports all FFT algorithms 
from scipy.io.wavfile import write
class FetchLabel():

    def get_emotion(self, file_path):
        item = file_path.split('/')[-1]
    
        emotion_map = {
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '01': 'neutral',
            '07': 'disgusted',
            '08': 'surprised'
        }
    
        # Check for the first letter-based mapping
        first_letter_map = {
            'a': 'angry',
            'f': 'fearful',
            'h': 'happy',
            'n': 'neutral',
            'sa': 'sad',
            'd': 'disgusted',
            'su': 'surprised'
        }
    
        if item[:2] in first_letter_map:
            return f"male_{first_letter_map[item[:2]]}"
        if item[:1] in first_letter_map:
            return f"male_{first_letter_map[item[:1]]}"
    
        emotion = emotion_map.get(item[6:-16])
        if emotion:
            gender = 'female' if int(item[18:-4]) % 2 == 0 else 'male'
            return f"{gender}_{emotion}"
    
        return None

label = FetchLabel()
        
# ! unzip /content/drive/MyDrive/output_folder_train.zip -d /content/drive/MyDrive/output_folder_train
# ! unzip /content/drive/MyDrive/speech-emotion-recognition-ravdess-data.zip -d /content/drive/MyDrive/
# rm /content/drive/MyDrive/output_folder_train/surprised/*
# Global var for directories
# AUDIO_FOLDER = "/content/drive/MyDrive/Actor_*/"
# OUTPUT_FOLDER_TRAIN = "/content/drive/MyDrive/output_folder_train/"
# OUTPUT_FOLDER_TEST = "/content/drive/MyDrive/output_folder_test"

# Plotting audio file
# Import one audio file with librosa
data, sampling_rate = librosa.load('/content/drive/MyDrive/Actor_10/03-01-01-01-01-01-10.wav')
plt.figure(figsize=(40, 5))                           # Shape of audio figure
librosa.display.waveplot(data, sr=sampling_rate)      # Show audio
# Load in audio file
y, sr = librosa.load('/content/drive/MyDrive/Actor_10/03-01-01-01-01-01-10.wav')
yt,_=librosa.effects.trim(y)                          # Trim leading and trailing silence from an audio signal.
# Converting the sound clips into a melspectogram with librosa
# A mel spectrogram is a spectrogram where the frequencies are converted to the mel scale
audio_spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=100)

# Convert a power spectrogram (amplitude squared) to decibel (dB) units with power_to_db
audio_spectogram = librosa.power_to_db(audio_spectogram, ref=np.max)

# Display the spectrogram with specshow
librosa.display.specshow(audio_spectogram, y_axis='mel', fmax=20000, x_axis='time')
# Extract features from audio using librosa
for actor in glob.glob(AUDIO_FOLDER):               # Loop through each actor in the data set 
  for name in glob.glob(actor +'/*'):               # Go through each audio file in each of the actors datasets
    print(name[-18:-16])                            # Sanity check: Check for name of file
    emotion = label.get_emotion(name[-24:])         # From audio file naming convention get the emotion of the data
    print(emotion)                                  # Sanity check: Check emotion name
# Dictionary with the numerical value that corresponds to each emotion
dicts={'01' : 'neutral', '02' : 'calm', '03' : 'happy', '04' : 'sad', '05' : 'angry', '06' : 'fearful', '07' : 'disgust', '08' : 'surprised'}

img = plt.imread('/content/drive/MyDrive/output_folder_train/angry/000001.jpg')   
plt.imshow(img)
from fastai.vision import *
train_path = Path("/content/drive/MyDrive/output_folder_train")
valid_path = Path("/content/drive/MyDrive/output_folder_train")
main_path = Path('/content/drive/MyDrive/Actor_*/')

# List all sentiment groups 
train_path.ls()

# Create from imagenet style dataset in path with train and valid subfolders (or provide valid_pct)
# from fastbook import *

dls = ImageDataLoaders.from_folder(train_path, valid_pct=0.2, seed=42, num_workers=0)
# dls.valid_ds.items[:10]

# Showcase the sentiment categories 
dls.vocab
# See what a sample of a batch looks like
# dls.show_batch(figsize=(7,8))
# This method creates a Learner object from the data object and model inferred from it with the backbone given in base_arch.
# ResNet-34 Pre-trained Model for PyTorch
learn = cnn_learner(dls, models.resnet34, loss_func=CrossEntropyLossFlat(), metrics=accuracy)

# Find optimum learning rate (the steepest point)
# lr_min, lr_steep = learn.lr_find()
# print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")

lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

# Sanity Check
print('learn.data.vocab', learn.dls.vocab)
# Train (fit) using the optimum learning rate
lr_min =lrs.minimum
lr_steep= lrs.steep
learn.fit(5, float(f"{lr_steep:.2e}"))
print(lr_min)
print(lr_steep)
# Plotting The losses for training and validation
learn.show_results()
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(dls.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learn.freeze()
learn.export('/content/drive/MyDrive/ser_saved_models/speech_1.pkl')
