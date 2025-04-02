# API Documentation

## Overview

The Activity Recommendation System provides a set of Python classes and functions for emotion recognition and activity recommendations. This document describes the main components and their usage.

## Face Recognition Module

### `FaceRecognition` Class

```python
class FaceRecognition:
    def __init__(self, model_path: str, cascade_path: str)
```

Initializes the face recognition system with paths to the model and cascade classifier files.

#### Parameters:
- `model_path`: Path to the pre-trained emotion recognition model
- `cascade_path`: Path to the face cascade classifier

#### Methods:

##### `load_model()`
Loads the pre-trained emotion recognition model.

##### `load_cascade()`
Loads the face cascade classifier.

##### `preprocess_image(image_path: Path) -> Tuple[np.ndarray, np.ndarray]`
Preprocesses an image for face detection and emotion recognition.

##### `detect_faces(blob: np.ndarray, image: np.ndarray) -> list`
Detects faces in the preprocessed image.

##### `predict_emotion(image_path: Path) -> Tuple[str, float]`
Predicts emotion from an image.

##### `save_results(emotion: str, confidence: float) -> None`
Saves the prediction results.

##### `cleanup() -> None`
Cleans up temporary files.

##### `run(image_path: Path) -> None`
Runs the complete face recognition process.

## Speech Recognition Module

### `SpeechEmotionRecognizer` Class

```python
class SpeechEmotionRecognizer:
    def __init__(self, model_path: str, audio_path: str)
```

Initializes the speech emotion recognition system with paths to the model and audio file.

#### Parameters:
- `model_path`: Path to the pre-trained model
- `audio_path`: Path to the audio file to process

#### Methods:

##### `load_model()`
Loads the pre-trained model for emotion recognition.

##### `process_audio() -> np.ndarray`
Processes the audio file and converts it into a mel spectrogram.

##### `save_spectrogram(audio_spectogram: np.ndarray) -> Path`
Saves the mel spectrogram as an image file.

##### `predict_emotion(image_path: Path) -> Tuple[str, np.ndarray, np.ndarray]`
Predicts emotion from the spectrogram image.

##### `save_results(text: str, img: np.ndarray) -> None`
Saves the predicted emotion and the spectrogram image.

##### `cleanup() -> None`
Cleans up temporary files.

##### `run() -> None`
Runs the complete speech emotion recognition process.

## Main Application

### `ActivityRecommendationSystem` Class

```python
class ActivityRecommendationSystem:
    def __init__(self)
```

Initializes the activity recommendation system.

#### Methods:

##### `load_activities() -> Dict[str, List[str]]`
Loads activity recommendations from the configuration file.

##### `get_recommendations(emotion: str) -> List[str]`
Gets activity recommendations based on the detected emotion.

##### `process_face_emotion(uploaded_file) -> Optional[str]`
Processes facial emotion from an uploaded image.

##### `process_speech_emotion(uploaded_file) -> Optional[str]`
Processes speech emotion from an uploaded audio file.

## Configuration

### `Config` Class

```python
class Config:
    def __init__(self)
```

Manages application configuration.

#### Properties:
- `models_dir`: Directory containing model files
- `data_dir`: Directory for data files
- `temp_dir`: Directory for temporary files
- `logs_dir`: Directory for log files
- `activities_path`: Path to the activities configuration file

## Usage Examples

### Face Emotion Recognition

```python
from src.recognition.face_recognition import FaceRecognition
from pathlib import Path

# Initialize the face recognizer
recognizer = FaceRecognition(
    "path/to/model",
    "path/to/cascade.xml"
)

# Process an image
image_path = Path("path/to/image.jpg")
recognizer.run(image_path)
```

### Speech Emotion Recognition

```python
from src.recognition.speech_recognition import SpeechEmotionRecognizer
from pathlib import Path

# Initialize the speech recognizer
recognizer = SpeechEmotionRecognizer(
    "path/to/model.pkl",
    "path/to/audio.wav"
)

# Process the audio
recognizer.run()
```

### Activity Recommendations

```python
from src.app import ActivityRecommendationSystem

# Initialize the system
system = ActivityRecommendationSystem()

# Get recommendations for an emotion
recommendations = system.get_recommendations("happy")
for rec in recommendations:
    print(rec)
```

## Error Handling

All methods include proper error handling and logging. Common exceptions include:

- `FileNotFoundError`: When required files are not found
- `ValueError`: When input data is invalid
- `RuntimeError`: When processing fails

## Logging

The system uses a hierarchical logging system with three main loggers:

- `app_logger`: For application-level events
- `model_logger`: For model-related events
- `recognition_logger`: For recognition-related events

Log files are stored in the configured logs directory with timestamps.
