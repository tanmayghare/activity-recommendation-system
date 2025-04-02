import pytest
from pathlib import Path
import numpy as np
from src.recognition.face_recognition import FaceRecognition
from configs.settings import config

@pytest.fixture
def face_recognizer():
    """Create a FaceRecognition instance for testing."""
    model_path = Path(config.models_dir) / "face"
    cascade_path = Path(config.models_dir) / "haarcascade_frontalface_default.xml"
    return FaceRecognition(str(model_path), str(cascade_path))

@pytest.fixture
def sample_image(tmp_path):
    """Create a sample image for testing."""
    image_path = tmp_path / "test.jpg"
    # Create a simple test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 255  # White square in the middle
    import cv2
    cv2.imwrite(str(image_path), img)
    return image_path

def test_initialization(face_recognizer):
    """Test FaceRecognition initialization."""
    assert face_recognizer is not None
    assert face_recognizer.model is not None
    assert face_recognizer.face_cascade is not None

def test_load_model_invalid_path():
    """Test model loading with invalid path."""
    with pytest.raises(FileNotFoundError):
        FaceRecognition("invalid/path", "invalid/path")

def test_preprocess_image(face_recognizer, sample_image):
    """Test image preprocessing."""
    blob, image = face_recognizer.preprocess_image(sample_image)
    assert blob is not None
    assert image is not None
    assert isinstance(blob, np.ndarray)
    assert isinstance(image, np.ndarray)

def test_preprocess_image_invalid_path(face_recognizer):
    """Test preprocessing with invalid image path."""
    with pytest.raises(FileNotFoundError):
        face_recognizer.preprocess_image(Path("invalid/path.jpg"))

def test_detect_faces(face_recognizer, sample_image):
    """Test face detection."""
    blob, image = face_recognizer.preprocess_image(sample_image)
    faces = face_recognizer.detect_faces(blob, image)
    assert isinstance(faces, list)

def test_predict_emotion(face_recognizer, sample_image):
    """Test emotion prediction."""
    emotion, confidence = face_recognizer.predict_emotion(sample_image)
    assert isinstance(emotion, str)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

def test_save_results(face_recognizer, tmp_path):
    """Test saving results."""
    emotion = "happy"
    confidence = 0.95
    face_recognizer.save_results(emotion, confidence)
    
    result_path = Path(config.data_dir) / "results" / "FinalFaceEmotion.txt"
    assert result_path.exists()
    
    with open(result_path) as f:
        content = f.read()
        assert emotion.capitalize() in content
        assert str(confidence) in content

def test_cleanup(face_recognizer, tmp_path):
    """Test cleanup of temporary files."""
    # Create some temporary files
    temp_dir = face_recognizer.temp_dir
    for i in range(3):
        (temp_dir / f"test_{i}.jpg").touch()
    
    face_recognizer.cleanup()
    
    # Check that files are removed
    assert len(list(temp_dir.glob("*.jpg"))) == 0

def test_run_complete_process(face_recognizer, sample_image):
    """Test the complete face recognition process."""
    face_recognizer.run(sample_image)
    
    # Check that results were saved
    result_path = Path(config.data_dir) / "results" / "FinalFaceEmotion.txt"
    assert result_path.exists()
    
    # Check that temporary files were cleaned up
    assert len(list(face_recognizer.temp_dir.glob("*.jpg"))) == 0
