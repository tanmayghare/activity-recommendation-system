import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import soundfile as sf

class FetchLabel:
    def get_emotion(self, file_path):
        """
        Extracts emotion from the RAVDESS audio file name.
        RAVDESS file naming convention: [Modality]-[VocalChannel]-[Emotion]-[EmotionalIntensity]-[Statement]-[Repetition]-[Actor]
        """
        try:
            # Extract emotion code from filename
            emotion_code = int(file_path.split('-')[2])
            emotion_map = {
                1: "neutral",
                2: "calm",
                3: "happy",
                4: "sad",
                5: "angry",
                6: "fearful",
                7: "disgust",
                8: "surprised"
            }
            return emotion_map.get(emotion_code, "unknown")
        except:
            return "unknown"

class SERModel:
    def __init__(self, audio_folder=None, output_folder_train=None, output_folder_test=None):
        """
        Initialize the SER model with paths and required components.
        """
        self.audio_folder = audio_folder or "data/datasets/ravdess_ser_dataset"
        self.output_folder_train = output_folder_train or "data/datasets/ser/train"
        self.output_folder_test = output_folder_test or "data/datasets/ser/test"
        self.label = FetchLabel()
        self.scaler = StandardScaler()
        self.model = SVC(kernel='rbf', probability=True)
        self.features = []
        self.labels = []

    def extract_features(self, file_path):
        """
        Extracts audio features from the given file.
        Returns a numpy array of features.
        """
        try:
            # Load audio file
            y, sr = librosa.load(file_path, duration=3)  # Load first 3 seconds
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            
            # Calculate statistics of features
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            chroma_mean = np.mean(chroma, axis=1)
            mel_mean = np.mean(mel, axis=1)
            
            # Combine all features
            features = np.concatenate([mfccs_mean, mfccs_std, chroma_mean, mel_mean])
            return features
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def plot_audio_file(self, file_path):
        """
        Plots the waveform and mel spectrogram of the audio file.
        """
        try:
            y, sr = librosa.load(file_path)
            
            plt.figure(figsize=(12, 8))
            
            # Plot waveform
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title('Waveform')
            
            # Plot mel spectrogram
            plt.subplot(2, 1, 2)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting {file_path}: {str(e)}")

    def process_audio_files(self):
        """
        Processes all audio files in the specified folder and extracts features.
        """
        self.features = []
        self.labels = []
        
        # Walk through all actor directories
        for root, _, files in os.walk(self.audio_folder):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    features = self.extract_features(file_path)
                    if features is not None:
                        self.features.append(features)
                        self.labels.append(self.label.get_emotion(file))

    def train_model(self):
        """
        Trains the model using the extracted features.
        """
        if not self.features or not self.labels:
            print("No features or labels available. Please run process_audio_files first.")
            return

        # Convert to numpy arrays
        X = np.array(self.features)
        y = np.array(self.labels)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save the model and scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/ser_model.joblib')
        joblib.dump(self.scaler, 'models/ser_scaler.joblib')
        print("\nModel saved to models/ser_model.joblib")

    def predict_emotion(self, audio_file):
        """
        Predicts the emotion of a given audio file.
        """
        try:
            # Extract features
            features = self.extract_features(audio_file)
            if features is None:
                return "Error processing audio file"

            # Scale features
            features_scaled = self.scaler.transform([features])

            # Predict emotion
            emotion = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            return {
                'emotion': emotion,
                'probabilities': dict(zip(self.model.classes_, probabilities))
            }
        except Exception as e:
            return f"Error predicting emotion: {str(e)}"

if __name__ == "__main__":
    # Initialize the SER model
    ser_model = SERModel()
    
    # Process audio files and extract features
    print("Processing audio files...")
    ser_model.process_audio_files()
    
    # Train the model
    print("\nTraining model...")
    ser_model.train_model()
    
    # Example prediction
    print("\nTesting prediction on a sample file...")
    sample_file = "data/datasets/ravdess_ser_dataset/Actor_01/03-01-01-01-01-01-01.wav"
    result = ser_model.predict_emotion(sample_file)
    print(f"Prediction result: {result}")