#!/usr/bin/env python3
"""
Script to train the facial and speech emotion recognition models.
"""

import os
import sys
import logging
from pathlib import Path
import argparse

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.models.fer_model import FERModel
from src.models.ser_model import SERModel
from configs.settings import config

# Set up logger
logger = setup_logger(__name__)

def train_fer_model():
    """Train the facial expression recognition model."""
    try:
        logger.info("Starting FER model training...")
        
        # Check if dataset directory exists
        if not Path("FER-2013").exists():
            logger.warning("FER-2013 dataset directory not found")
            logger.info("Please ensure the FER-2013 dataset is in the project root directory")
            logger.info("Expected structure: FER-2013/train/<emotion>/<image_files> and FER-2013/test/<emotion>/<image_files>")
            return False
        
        # Initialize and train the model
        fer_model = FERModel(
            data_path="FER-2013",
            pic_size=config.fer_config.input_size[0],
            batch_size=config.fer_config.batch_size,
            epochs=config.fer_config.epochs
        )
        
        # Visualize data
        logger.info("Visualizing training data...")
        fer_model.visualize_data()
        
        # Generate data
        logger.info("Generating data...")
        train_generator, validation_generator, test_generator = fer_model.data_generator()
        
        # Compile model
        logger.info("Compiling model...")
        fer_model.compile_model()
        
        # Train model
        logger.info("Training model...")
        history = fer_model.train_model(train_generator, validation_generator)
        
        # Evaluate model
        logger.info("Evaluating model...")
        test_loss, test_accuracy = fer_model.evaluate_model(test_generator)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Save model
        logger.info("Saving model...")
        model_path = Path(config.models_dir) / "face"
        model_path.mkdir(exist_ok=True)
        fer_model.save_model(str(model_path))
        
        logger.info("FER model training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error training FER model: {str(e)}")
        return False

def train_ser_model():
    """Train the speech emotion recognition model."""
    try:
        logger.info("Starting SER model training...")
        
        # Check if dataset directories exist
        if not config.ser_config.train_data_dir.exists():
            logger.warning(f"Training dataset directory not found: {config.ser_config.train_data_dir}")
            logger.info("Please download the dataset and place it in the correct location.")
            logger.info("Expected structure: {config.ser_config.train_data_dir}/<emotion>/<audio_files>")
            return False
            
        if not config.ser_config.val_data_dir.exists():
            logger.warning(f"Validation dataset directory not found: {config.ser_config.val_data_dir}")
            logger.info("Please download the dataset and place it in the correct location.")
            logger.info("Expected structure: {config.ser_config.val_data_dir}/<emotion>/<audio_files>")
            return False
        
        # Initialize and train the model
        ser_model = SERModel(
            audio_folder=str(config.ser_dataset_dir),
            output_folder_train=str(config.ser_config.train_data_dir),
            output_folder_test=str(config.ser_config.val_data_dir)
        )
        
        # Train model
        logger.info("Training model...")
        ser_model.train_model()
        
        logger.info("SER model training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error training SER model: {str(e)}")
        return False

def main():
    """Main function to train the models."""
    parser = argparse.ArgumentParser(description="Train emotion recognition models")
    parser.add_argument("--fer", action="store_true", help="Train facial expression recognition model")
    parser.add_argument("--ser", action="store_true", help="Train speech emotion recognition model")
    parser.add_argument("--all", action="store_true", help="Train all models")
    
    args = parser.parse_args()
    
    if not (args.fer or args.ser or args.all):
        parser.print_help()
        return 1
    
    success = True
    
    if args.fer or args.all:
        if not train_fer_model():
            success = False
    
    if args.ser or args.all:
        if not train_ser_model():
            success = False
    
    if success:
        logger.info("All model training completed successfully")
        return 0
    else:
        logger.error("Some model training failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 