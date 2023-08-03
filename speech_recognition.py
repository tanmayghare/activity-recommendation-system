# Speech recognition:
if __name__ == '__main__':

  import torch
  import matplotlib.pyplot as plt                       # Allows you to plot things
  import librosa                                       # Python package for music and audio analysis
  import librosa.display                                # Allows you to display audio files 
  import os                                             # The OS module in Python provides a way of using operating system dependent functionality.
  import scipy.io.wavfile                               # Open a WAV files
  import numpy as np                                    # Used for working with arrays
  import pickle
  import cv2   
  import pathlib
  temp = pathlib.PosixPath
  pathlib.PosixPath = pathlib.WindowsPath

  import fastai
  from fastai import *                                 
  from fastai.vision.all import *
  from fastai.vision.data import ImageDataLoaders
  from fastai.tabular.all import *
  from fastai.text.all import *
  from fastai.vision.widgets import *

  model = load_learner('/speech_1.pkl')
  p = pathlib.Path('/audio.wav')

  count = 1
  y, sr = librosa.load(p)
  yt,_=librosa.effects.trim(y)

  # Converting the sound clips into a melspectogram with librosa
  # A mel spectrogram is a spectrogram where the frequencies are converted to the mel scale
  audio_spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=100)

  # Convert a power spectrogram (amplitude squared) to decibel (dB) units with power_to_db
  audio_spectogram = librosa.power_to_db(audio_spectogram, ref=np.max)

  # Display the spectrogram with specshow
  librosa.display.specshow(audio_spectogram, y_axis='mel', fmax=20000, x_axis='time')

  p = os.path.join('/', "{}.jpg".format(str(count)))
  plt.savefig(p)

  p1 ='C:/Users/ghare/Desktop/BE Project Codes G86/1.jpg'
  # Print one image from sorted array file
  img = plt.imread(p1)

  text2, _, probs = model.predict(img)
  print(probs)
  print(text2)

  file1 = open("FinalSpeechEmotion.txt", "w")
  file1.write(text2.capitalize())
  file1.close()
  cv2.imwrite("/speechemotion.jpg", img)
