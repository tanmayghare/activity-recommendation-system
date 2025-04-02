#!/usr/bin/env python3
"""
Script to download required models.
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Optional

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from configs.settings import config

# Set up logger
logger = setup_logger(__name__)

# Model URLs and their destinations
MODELS = {
    "face": {
        "url": "https://storage.googleapis.com/activity-recommendation-system/models/face/emotion_model.pkl",
        "path": "models/face/emotion_model.pkl"
    },
    "face_cascade": {
        "url": "https://storage.googleapis.com/activity-recommendation-system/models/face/haarcascade_frontalface_default.xml",
        "path": "models/face/haarcascade_frontalface_default.xml"
    },
    "speech": {
        "url": "https://storage.googleapis.com/activity-recommendation-system/models/speech/speech_emotion_model.pkl",
        "path": "models/speech/speech_emotion_model.pkl"
    }
}

def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from a URL to a destination path with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file to
        chunk_size: Size of chunks to download
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Create parent directories if they don't exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Get file size for progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                pbar.update(size)
                
        return True
    except Exception as e:
        logger.error(f"Failed to download {url} to {destination}: {str(e)}")
        return False

def verify_model(model_path: Path) -> bool:
    """
    Verify that a downloaded model file exists and has content.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        bool: True if model is valid, False otherwise
    """
    try:
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
            
        if model_path.stat().st_size == 0:
            logger.error(f"Model file is empty: {model_path}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Failed to verify model {model_path}: {str(e)}")
        return False

def main():
    """Main function to download and verify all required models."""
    logger.info("Starting model download process...")
    
    # Create models directory
    models_dir = Path(config.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and verify each model
    success = True
    for model_name, model_info in MODELS.items():
        logger.info(f"Processing {model_name} model...")
        
        model_path = models_dir / model_info["path"]
        
        # Skip if model already exists and is valid
        if verify_model(model_path):
            logger.info(f"{model_name} model already exists and is valid")
            continue
            
        # Download model
        logger.info(f"Downloading {model_name} model...")
        if not download_file(model_info["url"], model_path):
            success = False
            continue
            
        # Verify downloaded model
        if not verify_model(model_path):
            success = False
            continue
            
        logger.info(f"Successfully downloaded and verified {model_name} model")
    
    if success:
        logger.info("All models downloaded and verified successfully")
        return 0
    else:
        logger.error("Some models failed to download or verify")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 