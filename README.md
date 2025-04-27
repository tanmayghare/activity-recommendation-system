# Activity Recommendation System

A sophisticated system that provides personalized activity recommendations based on emotion detection from facial expressions and speech patterns.

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   git clone <repository-url>
   cd activity-recommendation-system
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train Models**
   ```bash
   # Train Facial Expression Recognition Model
   python src/models/fer_model.py
   
   # Train Speech Emotion Recognition Model
   python src/models/ser_model.py
   ```
   - Training progress will be displayed
   - Models will be saved in the `models/` directory
   - Note: The required datasets (FER-2013 and RAVDESS) are already included in the repository

3. **Run the Application**
   ```bash
   streamlit run src/app.py
   ```

## 🎯 Features

- **Real-time Emotion Detection**
  - Facial Expression Analysis
  - Speech Pattern Recognition
- **Smart Recommendations**
  - Personalized Activity Suggestions
  - Context-Aware Responses
- **User-Friendly Interface**
  - Simple Upload System
  - Instant Results

## 📁 Project Structure

```
activity-recommendation-system/
├── data/           # Dataset storage
│   └── datasets/   # FER-2013 and RAVDESS datasets
├── models/         # Trained models
├── src/            # Source code
│   ├── models/     # ML models
│   ├── recommendation/  # LLM integration
│   └── utils/      # Helper functions
└── requirements.txt
```

## 🔧 Prerequisites

- Python 3.8+
- Virtual environment
- GPU (recommended for training)
- 8GB+ RAM
- Sufficient disk space for models

## 📊 Model Architecture

### Facial Expression Recognition
- CNN-based architecture
- 7 emotion classes (angry, disgust, fear, happy, sad, surprise, neutral)
- Real-time processing
- Input: 48x48 grayscale images

### Speech Emotion Recognition
- SVM with RBF kernel
- 8 emotion classes
- Audio feature extraction (MFCCs, chroma features)
- Input: WAV format audio files

### Recommendation System
- LLM-based suggestions
- Context-aware responses
- Personalized activities

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.