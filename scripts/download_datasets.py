#!/usr/bin/env python3
"""
Script to download datasets for the Activity Recommendation System.
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
from typing import Dict, List, Optional, Union

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from configs.settings import config

# Set up logger
logger = setup_logger(__name__)

# Dataset URLs and their destinations
DATASETS = {
    "fer": {
        "name": "Facial Emotion Recognition Dataset",
        "url": "https://storage.googleapis.com/activity-recommendation-system/datasets/fer2013.zip",
        "path": "data/datasets/fer",
        "extract": True,
        "expected_files": ["train", "val"]
    },
    "ser": {
        "name": "Speech Emotion Recognition Dataset",
        "url": "https://storage.googleapis.com/activity-recommendation-system/datasets/ravdess.tar.gz",
        "path": "data/datasets/ser",
        "extract": True,
        "expected_files": ["Actor_01", "Actor_02"]
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

def extract_archive(archive_path: Path, extract_path: Path) -> bool:
    """
    Extract an archive file to the specified path.
    
    Args:
        archive_path: Path to the archive file
        extract_path: Path to extract to
        
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Create extraction directory if it doesn't exist
        extract_path.mkdir(parents=True, exist_ok=True)
        
        # Extract based on file extension
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif archive_path.suffix == '.tar.gz' or archive_path.suffix == '.tgz':
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_path)
        else:
            logger.error(f"Unsupported archive format: {archive_path.suffix}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Failed to extract {archive_path} to {extract_path}: {str(e)}")
        return False

def verify_dataset(dataset_path: Path, expected_files: List[str]) -> bool:
    """
    Verify that a dataset directory contains the expected files.
    
    Args:
        dataset_path: Path to the dataset directory
        expected_files: List of expected files or directories
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    try:
        if not dataset_path.exists():
            logger.error(f"Dataset directory not found: {dataset_path}")
            return False
            
        # Check for expected files
        for expected_file in expected_files:
            expected_path = dataset_path / expected_file
            if not expected_path.exists():
                logger.error(f"Expected file/directory not found: {expected_path}")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Failed to verify dataset {dataset_path}: {str(e)}")
        return False

def download_dataset(dataset_name: str, dataset_info: Dict) -> bool:
    """
    Download and extract a dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_info: Dataset information
        
    Returns:
        bool: True if download and extraction was successful, False otherwise
    """
    try:
        logger.info(f"Processing {dataset_info['name']}...")
        
        # Create dataset directory
        dataset_path = Path(dataset_info['path'])
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset already exists and is valid
        if verify_dataset(dataset_path, dataset_info['expected_files']):
            logger.info(f"{dataset_info['name']} already exists and is valid")
            return True
            
        # Download dataset
        archive_name = dataset_info['url'].split('/')[-1]
        archive_path = dataset_path / archive_name
        
        logger.info(f"Downloading {dataset_info['name']}...")
        if not download_file(dataset_info['url'], archive_path):
            return False
            
        # Extract dataset if needed
        if dataset_info['extract']:
            logger.info(f"Extracting {dataset_info['name']}...")
            if not extract_archive(archive_path, dataset_path):
                return False
                
            # Remove archive file after extraction
            archive_path.unlink()
            
        # Verify extracted dataset
        if not verify_dataset(dataset_path, dataset_info['expected_files']):
            logger.error(f"Extracted dataset is invalid: {dataset_path}")
            return False
            
        logger.info(f"Successfully downloaded and extracted {dataset_info['name']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {dataset_name}: {str(e)}")
        return False

def main():
    """Main function to download datasets."""
    parser = argparse.ArgumentParser(description="Download datasets for emotion recognition")
    parser.add_argument("--fer", action="store_true", help="Download facial emotion recognition dataset")
    parser.add_argument("--ser", action="store_true", help="Download speech emotion recognition dataset")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    
    args = parser.parse_args()
    
    if not (args.fer or args.ser or args.all):
        parser.print_help()
        return 1
    
    success = True
    
    if args.fer or args.all:
        if not download_dataset("fer", DATASETS["fer"]):
            success = False
    
    if args.ser or args.all:
        if not download_dataset("ser", DATASETS["ser"]):
            success = False
    
    if success:
        logger.info("All datasets downloaded successfully")
        return 0
    else:
        logger.error("Some datasets failed to download")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 