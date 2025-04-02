"""Configuration management for the Activity Recommendation System."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "tmp"
LOGS_DIR = BASE_DIR / "logs"

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    model_path: str
    input_size: tuple
    classes: List[str]
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001

@dataclass
class AppConfig:
    """Application-specific configuration."""
    allowed_image_types: List[str]
    allowed_audio_types: List[str]
    max_file_size: int  # in MB
    log_level: str
    log_file: Optional[str]

class Config:
    """Main configuration class."""
    def __init__(self):
        """Initialize configuration with default values."""
        self._create_directories()
        self._init_configs()

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [DATA_DIR, MODELS_DIR, TEMP_DIR, LOGS_DIR]:
            directory.mkdir(exist_ok=True)
            
        # Create subdirectories
        (DATA_DIR / "raw").mkdir(exist_ok=True)
        (DATA_DIR / "processed").mkdir(exist_ok=True)

    def _init_configs(self):
        """Initialize specific configurations."""
        self.fer_config = ModelConfig(
            model_path=str(MODELS_DIR / "fer_model.h5"),
            input_size=(48, 48),
            classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
            batch_size=32,
            epochs=50
        )

        self.ser_config = ModelConfig(
            model_path=str(MODELS_DIR / "ser_model.pkl"),
            input_size=(128, 128),
            classes=["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
            batch_size=16,
            epochs=30
        )

        self.app_config = AppConfig(
            allowed_image_types=["jpg", "jpeg", "png"],
            allowed_audio_types=["wav", "mp3"],
            max_file_size=10,  # 10MB
            log_level="INFO",
            log_file="app.log"
        )

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return DATA_DIR

    @property
    def temp_dir(self) -> Path:
        """Get temporary directory path."""
        return TEMP_DIR

    @property
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        return LOGS_DIR

    @property
    def face_cascade_path(self) -> str:
        """Get path to face cascade classifier."""
        return str(BASE_DIR / "config" / "haarcascade_frontalface_default.xml")

    @property
    def activities_file(self) -> str:
        """Get path to activities configuration file."""
        return str(DATA_DIR / "activities.json")

# Create global instance
config = Config() 