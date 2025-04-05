import os
from typing import Union, Dict, List
from pathlib import Path
import numpy as np
from src.models.fer_model import FERModel
from src.models.ser_model import SERModel
from src.recommendation.llm_recommender import LLMRecommender
from src.utils.logger import app_logger
import cv2
import joblib

class EmotionDetectionSystem:
    """Core system for emotion detection and activity recommendation."""
    
    def __init__(self):
        """Initialize the emotion detection system with Facial Expression Recognition, SER, and LLM recommender."""
        try:
            # Initialize models with default paths
            self.fer_model = FERModel(
                data_path="data/datasets/FER-2013",
                pic_size=48,
                batch_size=32,
                epochs=50
            )
            
            self.ser_model = SERModel(
                audio_folder="data/datasets/ravdess_ser_dataset",
                output_folder_train="data/datasets/ser/train",
                output_folder_test="data/datasets/ser/test"
            )
            
            self.llm_recommender = LLMRecommender()
            
            app_logger.info("EmotionDetectionSystem initialized successfully")
        except Exception as e:
            app_logger.error(f"Failed to initialize EmotionDetectionSystem: {str(e)}")
            raise

    def detect_emotion_from_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Detect emotion from an image using Facial Expression Recognition model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing emotion and confidence
        """
        try:
            # Load and preprocess image
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Resize image
            img = cv2.resize(img, (48, 48))
            
            # Normalize and reshape for model input
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=-1)
            
            # Predict emotion
            predictions = self.fer_model.model.predict(img)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            
            return {
                "emotion": self.fer_model.emotion_labels[emotion_idx].lower(),
                "confidence": confidence,
                "probabilities": dict(zip(
                    [label.lower() for label in self.fer_model.emotion_labels],
                    predictions[0].tolist()
                )),
                "source": "image"
            }
        except Exception as e:
            app_logger.error(f"Error detecting emotion from image: {str(e)}")
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "source": "image",
                "error": str(e)
            }

    def detect_emotion_from_audio(self, audio_path: Union[str, Path]) -> Dict:
        """
        Detect emotion from an audio file using SER model.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing emotion and confidence
        """
        try:
            result = self.ser_model.predict_emotion(str(audio_path))
            
            if isinstance(result, dict):
                return {
                    "emotion": result["emotion"],
                    "confidence": max(result["probabilities"].values()),
                    "probabilities": result["probabilities"],
                    "source": "audio"
                }
            else:
                return {
                    "emotion": "unknown",
                    "confidence": 0.0,
                    "source": "audio",
                    "error": result
                }
        except Exception as e:
            app_logger.error(f"Error detecting emotion from audio: {str(e)}")
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "source": "audio",
                "error": str(e)
            }

    def get_recommendations(self, emotion: str, num_recommendations: int = 5) -> List[str]:
        """
        Get activity recommendations based on detected emotion.
        
        Args:
            emotion: Detected emotion
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of activity recommendations
        """
        try:
            return self.llm_recommender.get_recommendations(emotion, num_recommendations)
        except Exception as e:
            app_logger.error(f"Error getting recommendations: {str(e)}")
            return [
                "Take a mindful walk in nature - helps reset your emotional state",
                "Practice deep breathing exercises - promotes relaxation and clarity",
                "Listen to calming music - helps regulate emotions",
                "Write in a journal - provides emotional release and perspective",
                "Engage in light physical activity - boosts mood and energy"
            ][:num_recommendations]

    def process_input(self, file_path: Union[str, Path]) -> Dict:
        """
        Process input file (image or audio) and return emotion detection results with recommendations.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Dictionary containing emotion detection results and recommendations
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file type and process accordingly
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                emotion_result = self.detect_emotion_from_image(file_path)
            elif file_path.suffix.lower() in ['.wav', '.mp3']:
                emotion_result = self.detect_emotion_from_audio(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # Get recommendations if emotion was successfully detected
            if emotion_result["emotion"] != "unknown":
                recommendations = self.get_recommendations(emotion_result["emotion"])
                emotion_result["recommendations"] = recommendations
            
            return emotion_result
            
        except Exception as e:
            app_logger.error(f"Error processing input: {str(e)}")
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "source": "unknown",
                "error": str(e)
            }

if __name__ == "__main__":
    # Example usage
    system = EmotionDetectionSystem()
    
    # Test with sample files
    test_image = "data/datasets/FER-2013/test/happy/example.jpg"
    test_audio = "data/datasets/ravdess_ser_dataset/Actor_01/03-01-01-01-01-01-01.wav"
    
    # Process image
    if os.path.exists(test_image):
        result = system.process_input(test_image)
        print("\nImage Processing Result:")
        print(result)
    
    # Process audio
    if os.path.exists(test_audio):
        result = system.process_input(test_audio)
        print("\nAudio Processing Result:")
        print(result) 