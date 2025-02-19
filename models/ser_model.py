import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from fastai.vision.all import *
from fastai.tabular.all import *
from fastai.text.all import *
from fastai.vision.widgets import *

class FetchLabel:
    def get_emotion(self, file_path):
        """
        Extracts emotion from the file path based on predefined mappings.
        """
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

class SERModel:
    def __init__(self, audio_folder, output_folder_train, output_folder_test):
        self.audio_folder = audio_folder
        self.output_folder_train = output_folder_train
        self.output_folder_test = output_folder_test
        self.label = FetchLabel()

    def plot_audio_file(self, file_path):
        """
        Plots the waveform of the audio file.
        """
        data, sampling_rate = librosa.load(file_path)
        plt.figure(figsize=(40, 5))
        librosa.display.waveplot(data, sr=sampling_rate)
        plt.show()

    def extract_features(self, file_path):
        """
        Extracts and displays the mel spectrogram of the audio file.
        """
        y, sr = librosa.load(file_path)
        yt, _ = librosa.effects.trim(y)
        audio_spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=100)
        audio_spectogram = librosa.power_to_db(audio_spectogram, ref=np.max)
        librosa.display.specshow(audio_spectogram, y_axis='mel', fmax=20000, x_axis='time')
        plt.show()

    def process_audio_files(self):
        """
        Processes all audio files in the specified folder and prints their emotions.
        """
        for actor in glob.glob(self.audio_folder):
            for name in glob.glob(actor + '/*'):
                emotion = self.label.get_emotion(name[-24:])
                print(emotion)

    def train_model(self):
        """
        Trains the model using the training data and saves the trained model.
        """
        # Load training data
        train_path = Path(self.output_folder_train)
        dls = ImageDataLoaders.from_folder(train_path, valid_pct=0.2, seed=42, num_workers=0)
        
        # Create a CNN learner with ResNet-34 architecture
        learn = cnn_learner(dls, models.resnet34, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
        
        # Find the optimal learning rate
        lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
        lr_steep = lrs.steep
        
        # Train the model
        learn.fit(5, float(f"{lr_steep:.2e}"))
        
        # Show training results
        learn.show_results()
        
        # Interpret the results
        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
        
        # Freeze the model and export it
        learn.freeze()
        learn.export('ser_saved_models/speech_1.pkl')

if __name__ == "__main__":
    audio_folder = "Actor_*/"
    output_folder_train = "output_folder_train"
    output_folder_test = "output_folder_test"
    
    # Initialize and run the SER model
    ser_model = SERModel(audio_folder, output_folder_train, output_folder_test)
    ser_model.plot_audio_file('Actor_10/03-01-01-01-01-01-10.wav')
    ser_model.extract_features('Actor_10/03-01-01-01-01-01-10.wav')
    ser_model.process_audio_files()
    ser_model.train_model()