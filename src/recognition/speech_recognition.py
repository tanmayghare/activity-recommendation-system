import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import numpy as np
import cv2
from pathlib import Path
from fastai.vision.all import load_learner
from typing import Tuple, Optional
from ..utils.logger import recognition_logger
from config.settings import config

class SpeechEmotionRecognizer:
    def __init__(self, model_path: str, audio_path: str):
        """Initialize the speech emotion recognizer.
        
        Args:
            model_path: Path to the pre-trained model
            audio_path: Path to the audio file to process
        """
        self.model_path = Path(model_path)
        self.audio_path = Path(audio_path)
        self.temp_dir = Path(config.temp_dir) / "speech"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.model = self.load_model()
        self.count = 1

    def load_model(self):
        """Load the pre-trained model for emotion recognition."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            recognition_logger.info(f"Loading model from {self.model_path}")
            return load_learner(self.model_path)
        except Exception as e:
            recognition_logger.error(f"Failed to load model: {str(e)}")
            raise

    def process_audio(self) -> np.ndarray:
        """Process the audio file and convert it into a mel spectrogram."""
        try:
            if not self.audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
            
            recognition_logger.info(f"Processing audio file: {self.audio_path}")
            y, sr = librosa.load(self.audio_path)
            yt, _ = librosa.effects.trim(y)
            audio_spectogram = librosa.feature.melspectrogram(y=yt, sr=sr, n_fft=1024, hop_length=100)
            audio_spectogram = librosa.power_to_db(audio_spectogram, ref=np.max)
            return audio_spectogram
        except Exception as e:
            recognition_logger.error(f"Failed to process audio: {str(e)}")
            raise

    def save_spectrogram(self, audio_spectogram: np.ndarray) -> Path:
        """Save the mel spectrogram as an image file."""
        try:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(audio_spectogram, y_axis='mel', fmax=20000, x_axis='time')
            output_path = self.temp_dir / f"spectrogram_{self.count}.jpg"
            plt.savefig(output_path)
            plt.close()
            recognition_logger.info(f"Saved spectrogram to {output_path}")
            return output_path
        except Exception as e:
            recognition_logger.error(f"Failed to save spectrogram: {str(e)}")
            raise

    def predict_emotion(self, image_path: Path) -> Tuple[str, np.ndarray, np.ndarray]:
        """Predict the emotion from the spectrogram image."""
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Spectrogram image not found: {image_path}")
            
            recognition_logger.info(f"Predicting emotion from {image_path}")
            img = plt.imread(str(image_path))
            text, _, probs = self.model.predict(img)
            return text, probs, img
        except Exception as e:
            recognition_logger.error(f"Failed to predict emotion: {str(e)}")
            raise

    def save_results(self, text: str, img: np.ndarray) -> None:
        """Save the predicted emotion and the spectrogram image."""
        try:
            results_dir = Path(config.data_dir) / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save emotion text
            emotion_path = results_dir / "FinalSpeechEmotion.txt"
            with open(emotion_path, "w") as file:
                file.write(text.capitalize())
            
            # Save spectrogram image
            image_path = results_dir / "speechemotion.jpg"
            cv2.imwrite(str(image_path), img)
            
            recognition_logger.info(f"Saved results to {results_dir}")
        except Exception as e:
            recognition_logger.error(f"Failed to save results: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            for file in self.temp_dir.glob("*.jpg"):
                file.unlink()
            recognition_logger.info("Cleaned up temporary files")
        except Exception as e:
            recognition_logger.warning(f"Failed to cleanup temporary files: {str(e)}")

    def run(self) -> None:
        """Run the complete process of emotion recognition."""
        try:
            audio_spectogram = self.process_audio()
            image_path = self.save_spectrogram(audio_spectogram)
            text, probs, img = self.predict_emotion(image_path)
            recognition_logger.info(f"Predicted emotion: {text} with probabilities: {probs}")
            self.save_results(text, img)
        except Exception as e:
            recognition_logger.error(f"Failed to run emotion recognition: {str(e)}")
            raise
        finally:
            self.cleanup()

if __name__ == '__main__':
    try:
        model_path = Path(config.models_dir) / "speech_1.pkl"
        audio_path = Path(config.data_dir) / "raw" / "audio.wav"
        recognizer = SpeechEmotionRecognizer(str(model_path), str(audio_path))
        recognizer.run()
    except Exception as e:
        recognition_logger.error(f"Application error: {str(e)}")
        raise
