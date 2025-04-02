import pytest
from pathlib import Path
import numpy as np
import librosa
from src.recognition.speech_recognition import SpeechEmotionRecognizer
from config.settings import config

@pytest.fixture
def speech_recognizer():
    """Create a SpeechEmotionRecognizer instance for testing."""
    model_path = Path(config.models_dir) / "speech_1.pkl"
    audio_path = Path(config.data_dir) / "raw" / "test.wav"
    return SpeechEmotionRecognizer(str(model_path), str(audio_path))

@pytest.fixture
def sample_audio(tmp_path):
    """Create a sample audio file for testing."""
    audio_path = tmp_path / "test.wav"
    # Create a simple test audio file
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    librosa.output.write_wav(str(audio_path), audio, sr)
    return audio_path

def test_initialization(speech_recognizer):
    """Test SpeechEmotionRecognizer initialization."""
    assert speech_recognizer is not None
    assert speech_recognizer.model is not None
    assert speech_recognizer.temp_dir.exists()

def test_load_model_invalid_path():
    """Test model loading with invalid path."""
    with pytest.raises(FileNotFoundError):
        SpeechEmotionRecognizer("invalid/path", "invalid/path")

def test_process_audio(speech_recognizer, sample_audio):
    """Test audio processing."""
    spectrogram = speech_recognizer.process_audio()
    assert spectrogram is not None
    assert isinstance(spectrogram, np.ndarray)

def test_process_audio_invalid_path(speech_recognizer):
    """Test processing with invalid audio path."""
    with pytest.raises(FileNotFoundError):
        speech_recognizer.process_audio(Path("invalid/path.wav"))

def test_save_spectrogram(speech_recognizer, sample_audio):
    """Test spectrogram saving."""
    spectrogram = speech_recognizer.process_audio()
    output_path = speech_recognizer.save_spectrogram(spectrogram)
    assert output_path.exists()
    assert output_path.suffix == ".jpg"

def test_predict_emotion(speech_recognizer, sample_audio):
    """Test emotion prediction."""
    spectrogram = speech_recognizer.process_audio()
    image_path = speech_recognizer.save_spectrogram(spectrogram)
    text, probs, img = speech_recognizer.predict_emotion(image_path)
    assert isinstance(text, str)
    assert isinstance(probs, np.ndarray)
    assert isinstance(img, np.ndarray)

def test_save_results(speech_recognizer, tmp_path):
    """Test saving results."""
    text = "happy"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    speech_recognizer.save_results(text, img)
    
    result_path = Path(config.data_dir) / "results" / "FinalSpeechEmotion.txt"
    assert result_path.exists()
    
    with open(result_path) as f:
        content = f.read()
        assert text.capitalize() in content

def test_cleanup(speech_recognizer, tmp_path):
    """Test cleanup of temporary files."""
    # Create some temporary files
    temp_dir = speech_recognizer.temp_dir
    for i in range(3):
        (temp_dir / f"test_{i}.jpg").touch()
    
    speech_recognizer.cleanup()
    
    # Check that files are removed
    assert len(list(temp_dir.glob("*.jpg"))) == 0

def test_run_complete_process(speech_recognizer, sample_audio):
    """Test the complete speech recognition process."""
    speech_recognizer.run()
    
    # Check that results were saved
    result_path = Path(config.data_dir) / "results" / "FinalSpeechEmotion.txt"
    assert result_path.exists()
    
    # Check that temporary files were cleaned up
    assert len(list(speech_recognizer.temp_dir.glob("*.jpg"))) == 0
