import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from ..utils.logger import recognition_logger
from configs.settings import config

class FaceRecognition:
    def __init__(self, model_path: str, cascade_path: str):
        """Initialize the face recognition system.
        
        Args:
            model_path: Path to the pre-trained emotion recognition model
            cascade_path: Path to the face cascade classifier
        """
        self.model_path = Path(model_path)
        self.cascade_path = Path(cascade_path)
        self.temp_dir = Path(config.temp_dir) / "face"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.model = self.load_model()
        self.face_cascade = self.load_cascade()

    def load_model(self):
        """Load the pre-trained emotion recognition model."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            recognition_logger.info(f"Loading model from {self.model_path}")
            return cv2.dnn.readNetFromCaffe(
                str(self.model_path / "deploy.prototxt"),
                str(self.model_path / "res10_300x300_ssd_iter_140000.caffemodel")
            )
        except Exception as e:
            recognition_logger.error(f"Failed to load model: {str(e)}")
            raise

    def load_cascade(self):
        """Load the face cascade classifier."""
        try:
            if not self.cascade_path.exists():
                raise FileNotFoundError(f"Cascade file not found: {self.cascade_path}")
            recognition_logger.info(f"Loading cascade from {self.cascade_path}")
            return cv2.CascadeClassifier(str(self.cascade_path))
        except Exception as e:
            recognition_logger.error(f"Failed to load cascade: {str(e)}")
            raise

    def preprocess_image(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the image for face detection and expression recognition.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (processed image, original image)
        """
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            recognition_logger.info(f"Processing image: {image_path}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Resize image for face detection
            height, width = image.shape[:2]
            max_size = 300
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                image = cv2.resize(image, None, fx=scale, fy=scale)
            
            # Convert to blob for face detection
            blob = cv2.dnn.blobFromImage(
                image, 1.0, (300, 300),
                (104.0, 177.0, 123.0)
            )
            
            return blob, image
        except Exception as e:
            recognition_logger.error(f"Failed to preprocess image: {str(e)}")
            raise

    def detect_faces(self, blob: np.ndarray, image: np.ndarray) -> list:
        """Detect faces in the image.
        
        Args:
            blob: Preprocessed image blob
            image: Original image
            
        Returns:
            List of detected face coordinates (x, y, w, h)
        """
        try:
            self.model.setInput(blob)
            detections = self.model.forward()
            
            faces = []
            height, width = image.shape[:2]
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    x, y, w, h = box.astype(int)
                    faces.append((x, y, w, h))
            
            recognition_logger.info(f"Detected {len(faces)} faces in image")
            return faces
        except Exception as e:
            recognition_logger.error(f"Failed to detect faces: {str(e)}")
            raise

    def predict_emotion(self, image_path: Path) -> Tuple[str, float]:
        """Predict emotion from the image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (predicted emotion, confidence score)
        """
        try:
            blob, image = self.preprocess_image(image_path)
            faces = self.detect_faces(blob, image)
            
            if not faces:
                raise ValueError("No faces detected in the image")
            
            # Process the first detected face
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]
            
            # Save the cropped face for debugging
            debug_path = self.temp_dir / f"face_{Path(image_path).stem}.jpg"
            cv2.imwrite(str(debug_path), face)
            
            # TODO: Add actual emotion prediction logic here
            # For now, return a dummy prediction
            emotion = "neutral"
            confidence = 0.95
            
            recognition_logger.info(f"Predicted emotion: {emotion} with confidence: {confidence}")
            return emotion, confidence
        except Exception as e:
            recognition_logger.error(f"Failed to predict emotion: {str(e)}")
            raise

    def save_results(self, emotion: str, confidence: float) -> None:
        """Save the prediction results.
        
        Args:
            emotion: Predicted emotion
            confidence: Confidence score
        """
        try:
            results_dir = Path(config.data_dir) / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            result_path = results_dir / "FinalFacialExpression.txt"
            with open(result_path, "w") as file:
                file.write(f"{emotion.capitalize()}\nConfidence: {confidence:.2f}")
            
            recognition_logger.info(f"Saved results to {result_path}")
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

    def run(self, image_path: Path) -> None:
        """Run the complete face recognition process.
        
        Args:
            image_path: Path to the input image
        """
        try:
            emotion, confidence = self.predict_emotion(image_path)
            self.save_results(emotion, confidence)
        except Exception as e:
            recognition_logger.error(f"Failed to run face recognition: {str(e)}")
            raise
        finally:
            self.cleanup()

if __name__ == '__main__':
    try:
        model_path = Path(config.models_dir) / "fer_model.pkl"
        cascade_path = Path(config.config_dir) / "haarcascade_frontalface_default.xml"
        image_path = Path(config.data_dir) / "raw" / "test.jpg"
        
        recognizer = FaceRecognition(str(model_path), str(cascade_path))
        recognizer.run(image_path)
    except Exception as e:
        recognition_logger.error(f"Application error: {str(e)}")
        raise
