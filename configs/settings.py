"""Configuration management for the VibeCoach."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

# Base directories
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
TEMP_DIR: Path = BASE_DIR / "tmp"
LOGS_DIR: Path = BASE_DIR / "logs"

# Dataset paths
DATASET_DIR: Path = DATA_DIR / "datasets"
FER_DATASET_DIR: Path = DATASET_DIR / "FER-2013"
SER_DATASET_DIR: Path = DATASET_DIR / "ser"
FER_TRAIN_DIR: Path = FER_DATASET_DIR / "train"
FER_VAL_DIR: Path = FER_DATASET_DIR / "test"  # Using test set for validation
SER_TRAIN_DIR: Path = SER_DATASET_DIR / "train"
SER_VAL_DIR: Path = SER_DATASET_DIR / "val"

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    model_path: str
    input_size: tuple[int, int]
    classes: List[str]
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    train_data_dir: Optional[Path] = None
    val_data_dir: Optional[Path] = None

@dataclass
class LLMConfig:
    """Configuration for LLM-based recommendations."""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    temperature: float = 0.7
    max_tokens: int = 500
    num_recommendations: int = 5
    device: str = "auto"  # 'auto', 'cpu', 'cuda', or specific device like 'cuda:0'
    torch_dtype: str = "float16"  # 'float16' or 'float32'
    
    # Model alternatives for different resource constraints
    available_models: Dict[str, str] = {
        "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",   # Lightweight (1.1B parameters)
        "small": "facebook/opt-350m",                   # Very small (350M parameters)
        "medium": "facebook/opt-1.3b",                  # Medium size (1.3B parameters)
        "large": "stabilityai/stablelm-base-alpha-3b"   # Larger model (3B parameters)
    }

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
    def __init__(self) -> None:
        """Initialize configuration with default values."""
        self._create_directories()
        self._init_configs()
        self._llm_config = LLMConfig()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [DATA_DIR, MODELS_DIR, TEMP_DIR, LOGS_DIR, DATASET_DIR, 
                         FER_DATASET_DIR, SER_DATASET_DIR, 
                         FER_TRAIN_DIR, FER_VAL_DIR, 
                         SER_TRAIN_DIR, SER_VAL_DIR]:
            directory.mkdir(exist_ok=True)
            
        # Create results directory for model outputs
        (DATA_DIR / "results").mkdir(exist_ok=True)

    def _init_configs(self) -> None:
        """Initialize specific configurations."""
        self.fer_config = ModelConfig(
            model_path=str(MODELS_DIR / "fer_model.pkl"),
            input_size=(48, 48),
            classes=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
            batch_size=32,
            epochs=50,
            train_data_dir=FER_TRAIN_DIR,
            val_data_dir=FER_VAL_DIR
        )

        self.ser_config = ModelConfig(
            model_path=str(MODELS_DIR / "ser_model.pkl"),
            input_size=(128, 128),
            classes=["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
            batch_size=16,
            epochs=30,
            train_data_dir=SER_TRAIN_DIR,
            val_data_dir=SER_VAL_DIR
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
        return str(MODELS_DIR / "face" / "haarcascade_frontalface_default.xml")

    @property
    def fer_dataset_dir(self) -> Path:
        """Get path to facial expression recognition dataset directory."""
        return FER_DATASET_DIR
        
    @property
    def ser_dataset_dir(self) -> Path:
        """Get path to speech emotion recognition dataset directory."""
        return SER_DATASET_DIR

    @property
    def llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return self._llm_config

# Create global instance
config = Config() 