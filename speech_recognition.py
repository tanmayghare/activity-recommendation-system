import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import numpy as np
import cv2
import pathlib
from fastai.vision.all import load_learner

class SpeechEmotionRecognizer:
    def __init__(self, model_path, audio_path):
        self.model_path = model_path
        self.audio_path = audio_path
        self.model = self.load_model()
        self.count = 1

    def load_model(self):
        # Load the pre-trained model for emotion recognition
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        return load_learner(self.model_path)

    def process_audio(self):
        # Process the audio file and convert it into a mel spectrogram
        y, sr = librosa.load(self.audio_path)
        yt, _ = librosa.effects.trim(y)
        audio_spectogram = librosa.feature.melspectrogram(y=yt, sr=sr, n_fft=1024, hop_length=100)
        audio_spectogram = librosa.power_to_db(audio_spectogram, ref=np.max)
        return audio_spectogram

    def save_spectrogram(self, audio_spectogram):
        # Save the mel spectrogram as an image file
        librosa.display.specshow(audio_spectogram, y_axis='mel', fmax=20000, x_axis='time')
        output_path = os.path.join('/', f"{self.count}.jpg")
        plt.savefig(output_path)
        return output_path

    def predict_emotion(self, image_path):
        # Predict the emotion from the spectrogram image
        img = plt.imread(image_path)
        text2, _, probs = self.model.predict(img)
        return text2, probs, img

    def save_results(self, text, img):
        # Save the predicted emotion and the spectrogram image
        with open("FinalSpeechEmotion.txt", "w") as file:
            file.write(text.capitalize())
        cv2.imwrite("/speechemotion.jpg", img)

    def run(self):
        # Run the complete process of emotion recognition
        audio_spectogram = self.process_audio()
        image_path = self.save_spectrogram(audio_spectogram)
        text, probs, img = self.predict_emotion(image_path)
        print(probs)
        print(text)
        self.save_results(text, img)

if __name__ == '__main__':
    recognizer = SpeechEmotionRecognizer('/speech_1.pkl', '/audio.wav')
    recognizer.run()
