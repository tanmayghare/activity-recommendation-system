# Activity Recommendation System

A sophisticated machine learning system that provides personalized activity recommendations based on emotion detection from facial expressions and speech patterns.

## Features

- **Facial Emotion Recognition (FER)**
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
├── configs/
│   └── settings.py          # Configuration management
├── data/
│   ├── raw/                 # Raw input data
│   └── processed/           # Processed data
├── docs/
│   └── api.md              # API documentation
├── models/
│   ├── face/               # Face recognition models
│   └── speech/             # Speech recognition models
├── src/
│   ├── recognition/
│   │   ├── face_recognition.py
│   │   └── speech_recognition.py
│   └── utils/
│       └── logger.py
├── tests/
│   ├── test_fer_model.py
│   └── test_ser_model.py
├── app.py                  # Main application file
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Usage

1. Launch the application:
   ```bash
   streamlit run app.py
   ```

2. Access the web interface at `http://localhost:8501`

3. Process your input:
   - For facial emotion recognition: Upload a clear facial image
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

- OpenCV for computer vision capabilities
- Librosa for audio processing
- Streamlit for the web interface
- FastAI for the speech emotion recognition model
