import streamlit as st
from pathlib import Path
from typing import List, Optional
from src.recognition.face_recognition import FaceRecognition
from src.recognition.speech_recognition import SpeechEmotionRecognizer
from src.recommendation.llm_recommender import LLMRecommender
from src.utils.logger import app_logger
from configs.settings import config

class ActivityRecommendationSystem:
    def __init__(self):
        """Initialize the activity recommendation system with proper error handling."""
        try:
            self.face_recognizer = FaceRecognition(
                str(config.models_dir / "fer_model.pkl"),
                str(config.config_dir / "haarcascade_frontalface_default.xml")
            )
            self.speech_recognizer = SpeechEmotionRecognizer(
                str(config.models_dir / "ser_model.pkl"),
                str(config.data_dir / "raw" / "audio.wav")
            )
            self.llm_recommender = LLMRecommender()
            app_logger.info("ActivityRecommendationSystem initialized successfully")
        except Exception as e:
            app_logger.error(f"Failed to initialize ActivityRecommendationSystem: {str(e)}")
            raise

    def get_recommendations(self, emotion: str) -> List[str]:
        """Get activity recommendations based on the detected emotion with validation."""
        try:
            if not isinstance(emotion, str):
                raise ValueError("Emotion must be a string")
                
            emotion = emotion.lower().strip()
            if not emotion:
                raise ValueError("Emotion cannot be empty")
            
            recommendations = self.llm_recommender.get_recommendations(
                emotion,
                num_recommendations=config.llm_config.num_recommendations
            )
            app_logger.info(f"Generated {len(recommendations)} recommendations for {emotion}")
            return recommendations
        except Exception as e:
            app_logger.error(f"Failed to get recommendations: {str(e)}")
            return ["Error getting recommendations. Please try again."]

    def validate_image_file(self, file) -> bool:
        """Validate uploaded image file."""
        if file is None:
            return False
            
        # Check file size (max 5MB)
        if file.size > 5 * 1024 * 1024:
            st.error("Image file size must be less than 5MB")
            return False
            
        # Check file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if file.type not in allowed_types:
            st.error("Only JPEG and PNG images are supported")
            return False
            
        return True

    def validate_audio_file(self, file) -> bool:
        """Validate uploaded audio file."""
        if file is None:
            return False
            
        # Check file size (max 10MB)
        if file.size > 10 * 1024 * 1024:
            st.error("Audio file size must be less than 10MB")
            return False
            
        # Check file type
        if file.type != "audio/wav":
            st.error("Only WAV audio files are supported")
            return False
            
        return True

    def process_face_emotion(self, uploaded_file) -> Optional[str]:
        """Process facial emotion from the uploaded image with validation."""
        try:
            if not self.validate_image_file(uploaded_file):
                return None
            
            # Save the uploaded file
            image_path = Path(config.temp_dir) / "uploaded_face.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            app_logger.info(f"Processing face emotion from {image_path}")
            self.face_recognizer.run(image_path)
            
            # Read the result
            result_path = Path(config.data_dir) / "results" / "FinalFaceEmotion.txt"
            if not result_path.exists():
                raise FileNotFoundError(f"Result file not found: {result_path}")
                
            with open(result_path) as f:
                emotion = f.readline().strip().lower()
            
            app_logger.info(f"Detected face emotion: {emotion}")
            return emotion
        except Exception as e:
            app_logger.error(f"Failed to process face emotion: {str(e)}")
            st.error("Error processing facial emotion. Please try again.")
            return None

    def process_speech_emotion(self, uploaded_file) -> Optional[str]:
        """Process speech emotion from the uploaded audio file with validation."""
        try:
            if not self.validate_audio_file(uploaded_file):
                return None
            
            # Save the uploaded file
            audio_path = Path(config.temp_dir) / "uploaded_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            app_logger.info(f"Processing speech emotion from {audio_path}")
            self.speech_recognizer.run()
            
            # Read the result
            result_path = Path(config.data_dir) / "results" / "FinalSpeechEmotion.txt"
            if not result_path.exists():
                raise FileNotFoundError(f"Result file not found: {result_path}")
                
            with open(result_path) as f:
                emotion = f.readline().strip().lower()
            
            app_logger.info(f"Detected speech emotion: {emotion}")
            return emotion
        except Exception as e:
            app_logger.error(f"Failed to process speech emotion: {str(e)}")
            st.error("Error processing speech emotion. Please try again.")
            return None

def main():
    """Main application entry point with error handling."""
    try:
        st.set_page_config(
            page_title="Activity Recommendation System",
            page_icon="🎭",
            layout="wide"
        )
        
        st.title("Activity Recommendation System")
        st.write("Upload an image or audio file to get activity recommendations based on detected emotions.")
        
        system = ActivityRecommendationSystem()
        
        # File upload section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Facial Expression Recognition")
            face_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of a face (max 5MB)"
            )
        
        with col2:
            st.subheader("Speech Emotion Recognition")
            audio_file = st.file_uploader(
                "Upload an audio file",
                type=["wav"],
                help="Upload a WAV audio file (max 10MB)"
            )
        
        # Process button
        if st.button("Get Recommendations"):
            with st.spinner("Processing..."):
                face_emotion = system.process_face_emotion(face_file)
                speech_emotion = system.process_speech_emotion(audio_file)
                
                if face_emotion is None and speech_emotion is None:
                    st.warning("Please upload at least one valid file to get recommendations.")
                    return
                
                # Display results
                st.subheader("Detected Emotions")
                if face_emotion:
                    st.write(f"Facial Emotion: {face_emotion.capitalize()}")
                if speech_emotion:
                    st.write(f"Speech Emotion: {speech_emotion.capitalize()}")
                
                # Get and display recommendations
                st.subheader("Recommended Activities")
                emotions = [e for e in [face_emotion, speech_emotion] if e is not None]
                for emotion in emotions:
                    recommendations = system.get_recommendations(emotion)
                    st.write(f"For {emotion.capitalize()}:")
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
                    st.write("")
    
    except Exception as e:
        app_logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.")
        st.error("If the problem persists, please contact support.")

if __name__ == "__main__":
    main()
