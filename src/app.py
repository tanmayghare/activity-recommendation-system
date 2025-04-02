import streamlit as st
from pathlib import Path
import json
from typing import Dict, List, Optional
from .recognition.face_recognition import FaceRecognition
from .recognition.speech_recognition import SpeechEmotionRecognizer
from .utils.logger import app_logger
from config.settings import config

class ActivityRecommendationSystem:
    def __init__(self):
        """Initialize the activity recommendation system."""
        self.face_recognizer = FaceRecognition(
            str(config.models_dir / "face"),
            str(config.models_dir / "haarcascade_frontalface_default.xml")
        )
        self.speech_recognizer = SpeechEmotionRecognizer(
            str(config.models_dir / "speech_1.pkl"),
            str(config.data_dir / "raw" / "audio.wav")
        )
        self.activities = self.load_activities()

    def load_activities(self) -> Dict[str, List[str]]:
        """Load activity recommendations from the configuration file."""
        try:
            activities_path = Path(config.activities_path)
            if not activities_path.exists():
                raise FileNotFoundError(f"Activities file not found: {activities_path}")
            
            app_logger.info(f"Loading activities from {activities_path}")
            with open(activities_path) as f:
                return json.load(f)
        except Exception as e:
            app_logger.error(f"Failed to load activities: {str(e)}")
            raise

    def get_recommendations(self, emotion: str) -> List[str]:
        """Get activity recommendations based on the detected emotion."""
        try:
            emotion = emotion.lower()
            if emotion not in self.activities:
                app_logger.warning(f"Unknown emotion: {emotion}")
                return ["No specific recommendations available for this emotion."]
            
            recommendations = self.activities[emotion]
            app_logger.info(f"Found {len(recommendations)} recommendations for {emotion}")
            return recommendations
        except Exception as e:
            app_logger.error(f"Failed to get recommendations: {str(e)}")
            return ["Error getting recommendations. Please try again."]

    def process_face_emotion(self, uploaded_file) -> Optional[str]:
        """Process facial emotion from the uploaded image."""
        try:
            if uploaded_file is None:
                return None
            
            # Save the uploaded file
            image_path = Path(config.temp_dir) / "uploaded_face.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            app_logger.info(f"Processing face emotion from {image_path}")
            self.face_recognizer.run(image_path)
            
            # Read the result
            result_path = Path(config.data_dir) / "results" / "FinalFaceEmotion.txt"
            with open(result_path) as f:
                emotion = f.readline().strip().lower()
            
            app_logger.info(f"Detected face emotion: {emotion}")
            return emotion
        except Exception as e:
            app_logger.error(f"Failed to process face emotion: {str(e)}")
            st.error("Error processing facial emotion. Please try again.")
            return None

    def process_speech_emotion(self, uploaded_file) -> Optional[str]:
        """Process speech emotion from the uploaded audio file."""
        try:
            if uploaded_file is None:
                return None
            
            # Save the uploaded file
            audio_path = Path(config.temp_dir) / "uploaded_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            app_logger.info(f"Processing speech emotion from {audio_path}")
            self.speech_recognizer.run()
            
            # Read the result
            result_path = Path(config.data_dir) / "results" / "FinalSpeechEmotion.txt"
            with open(result_path) as f:
                emotion = f.readline().strip().lower()
            
            app_logger.info(f"Detected speech emotion: {emotion}")
            return emotion
        except Exception as e:
            app_logger.error(f"Failed to process speech emotion: {str(e)}")
            st.error("Error processing speech emotion. Please try again.")
            return None

def main():
    """Main application entry point."""
    try:
        st.title("Activity Recommendation System")
        st.write("Upload an image or audio file to get activity recommendations based on detected emotions.")
        
        system = ActivityRecommendationSystem()
        
        # File upload section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Facial Emotion Recognition")
            face_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of a face"
            )
        
        with col2:
            st.subheader("Speech Emotion Recognition")
            audio_file = st.file_uploader(
                "Upload an audio file",
                type=["wav"],
                help="Upload a WAV audio file"
            )
        
        # Process button
        if st.button("Get Recommendations"):
            face_emotion = system.process_face_emotion(face_file)
            speech_emotion = system.process_speech_emotion(audio_file)
            
            if face_emotion is None and speech_emotion is None:
                st.warning("Please upload at least one file to get recommendations.")
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
        st.error("An error occurred. Please try again later.")

if __name__ == "__main__":
    main()
