# Activity Recommendation System

A sophisticated system that provides personalized activity recommendations based on emotion detection from facial expressions and speech patterns.

## Features

- **Facial Expression Recognition (FER)**
  - Real-time emotion detection from facial expressions
  - Support for multiple emotion categories
  - High-accuracy processing pipeline

- **Speech Emotion Recognition (SER)**
  - Advanced speech pattern analysis for emotion detection
  - WAV audio file processing
  - Confidence score generation

- **Activity Recommendations**
  - Context-aware activity suggestions based on detected emotions
  - Multiple recommendation options per emotion
  - Intuitive web interface

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/activity-recommendation-system.git
   cd activity-recommendation-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required models:
   ```bash
   python scripts/download_models.py
   ```

## Project Structure

```
activity-recommendation-system/
├── .github/                # GitHub configuration and workflows
├── configs/
│   └── settings.py          # Configuration management
├── data/
│   ├── raw/                 # Raw input data
│   │   └── fer2013/         # FER-2013 dataset sourced from Kaggle
│   │       ├── train/       # Training images (emotion subfolders)
│   │       └── test/        # Test images (emotion subfolders)
│   ├── processed/           # Processed data
│   └── datasets/            # Additional dataset storage
├── docs/
│   └── api.md               # API documentation
├── models/                 # Trained model storage
├── scripts/
│   ├── download_datasets.py # Dataset download script
│   └── train_models.py      # Model training script
├── src/
│   ├── models/              # Model implementations
│   │   ├── fer_model.py     # Facial expression recognition
│   │   └── ser_model.py     # Speech emotion recognition
│   ├── recognition/
│   │   ├── face_recognition.py
│   │   └── speech_recognition.py
│   └── utils/
│       └── logger.py
├── tests/                  # Test suite
├── .gitignore
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Container orchestration
├── LICENSE
├── README.md
├── requirements.txt        # Production dependencies
└── requirements-dev.txt    # Development dependencies
```

## Usage

1. Launch the application:
   ```bash
   streamlit run app.py
   ```

2. Access the web interface at `http://localhost:8501`

3. Process your input:
   - For facial expression recognition: Upload a clear facial image
   - For speech emotion recognition: Upload a WAV audio file

4. Generate recommendations by clicking "Get Recommendations"

## Configuration

System configuration is managed through `configs/settings.py`:

- Model configurations and parameters
- Data directory paths
- Logging settings
- Application parameters

## Development

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Execute test suite:
   ```bash
   pytest tests/
   ```

3. Verify code quality:
   ```bash
   flake8 src/
   black src/
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/) - For image processing and facial detection capabilities
- [Librosa](https://librosa.org/) - For audio feature extraction and processing
- [NumPy](https://numpy.org/) - For numerical computing operations
- [FastAI](https://fast.ai/) - For deep learning model implementation in speech recognition
- [TensorFlow](https://www.tensorflow.org/) - For implementing the facial expression recognition model
- [PyTorch](https://pytorch.org/) - For deep learning model training and inference
- [Hugging Face Transformers](https://huggingface.co/transformers) - For open-source language models and tokenization
- [Matplotlib](https://matplotlib.org/) - For data visualization and image processing
- [Streamlit](https://streamlit.io/) - For creating our interactive web interface
- [Python](https://www.python.org/) - Core programming language