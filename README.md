# Activity Recommendation System

A machine learning-based system that recommends activities based on detected emotions from facial expressions and speech patterns.

## Features

- **Facial Emotion Recognition (FER)**
  - Detects emotions from facial expressions in images
  - Supports multiple emotion categories
  - Real-time processing

- **Speech Emotion Recognition (SER)**
  - Analyzes speech patterns to detect emotions
  - Processes WAV audio files
  - Provides confidence scores

- **Activity Recommendations**
  - Personalized activity suggestions based on detected emotions
  - Multiple recommendations per emotion
  - Easy-to-use web interface

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/activity-recommendation-system.git
   cd activity-recommendation-system
   ```

2. Create a virtual environment:
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
├── config/
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
│   ├── utils/
│   │   └── logger.py
│   └── app.py
├── tests/
│   ├── test_fer_model.py
│   └── test_ser_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Usage

1. Start the application:
   ```bash
   streamlit run src/app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload an image or audio file:
   - For facial emotion recognition: Upload a clear image of a face
   - For speech emotion recognition: Upload a WAV audio file

4. Click "Get Recommendations" to process the input and receive activity suggestions

## Configuration

The system can be configured through `config/settings.py`:

- Model paths and parameters
- Input/output directories
- Logging settings
- Application settings

## Development

1. Set up development environment:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run tests:
   ```bash
   pytest tests/
   ```

3. Check code style:
   ```bash
   flake8 src/
   black src/
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- Librosa for audio processing
- Streamlit for the web interface
- FastAI for the speech emotion recognition model
