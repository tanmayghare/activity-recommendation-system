import cv2
import json
import numpy as np
import streamlit as st
import random
from PIL import Image
from pydub import AudioSegment

class ActivityRecommendationSystem:
    def __init__(self, json_file_path):
        """Initialize the ActivityRecommendationSystem with the path to the JSON file containing activities."""
        self.json_file_path = json_file_path

    def activities(self, index, emotion):
        """Display activity recommendations based on the given emotion."""
        with open(self.json_file_path, 'r') as j:
            contents = json.loads(j.read())
            for keys in contents["activities"][index][emotion][0].values():
                for sub_key, sub_value in keys.items():
                    idx = random.randrange(0, len(sub_value))
                    url = f'[{sub_key}]({sub_value[idx]})'
                    st.write(f'''
                        <a href={sub_value[idx]} target="_blank">
                            <button style="background-color: rgb(255, 75, 75);
                                        border: none;
                                        color: white;
                                        padding: 10px 20px;
                                        text-align: center;
                                        text-decoration: none;
                                        display: inline-block;
                                        font-size: 16px;
                                        margin: 2px 2px;
                                        cursor: pointer;
                                        border-radius: 12px;">
                                {sub_key}
                            </button>
                        </a>
                        ''', unsafe_allow_html=True)

    def fer_upload_choice(self):
        """Handle the Face Expression Recognition (FER) upload or capture choice."""
        st.header("Face Expression Recognition:")
        fer_choice = st.radio("Please choose one option :", ("Upload Image", "Capture Image"))

        if fer_choice == "Upload Image":
            image_file = st.file_uploader("Upload Image :", type=["png", "jpg", "jpeg"])
            if image_file is not None:
                st.image(image_file, width=250)
                with open('image.jpg', "wb") as f:
                    f.write(image_file.getbuffer())
                st.success("File Saved")

        if fer_choice == "Capture Image":
            captured_image = st.camera_input("Take a picture :")
            if captured_image is not None:
                st.success("Image Captured")
                bytes_data = captured_image.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                with open('image.jpg', "wb") as f:
                    f.write(captured_image.getbuffer())

        run_fer_btn = st.button("Run FER")
        if run_fer_btn:
            try:
                exec(open('FaceRecognition.py').read())
            except:
                st.write("Please re-capture the image")
                return

            emotion_face_final = open('FinalFaceEmotion.txt', 'r').read()
            st.subheader("Predicted Emotion: " + emotion_face_final)
            image = Image.open('faceemotion.jpg')
            st.image(image)
            self.display_activities(emotion_face_final)

    def display_activities(self, emotion):
        """Display activities based on the detected emotion."""
        with open(self.json_file_path, 'r') as j:
            contents = json.loads(j.read())
            emotion_map = {
                "Happy": (0, "happy"),
                "Sad": (1, "sad"),
                "Angry": (2, "angry"),
                "Neutral": (3, "neutral"),
                "Surprise": (4, "surprised"),
                "Fearful": (5, "fearful"),
                "Disgust": (6, "disgust"),
                "Calm": (7, "calm")
            }
            if emotion in emotion_map:
                index, emotion_key = emotion_map[emotion]
                st.subheader(f"Suggesting you some activities you might find useful:")
                self.activities(index, emotion_key)

    def ser_upload_choice(self):
        """Handle the Speech Emotion Recognition (SER) upload choice."""
        st.header("Speech Emotion Recognition:")
        audio_file = st.file_uploader("Upload Audio Recording :", type=["wav"])
        if audio_file is not None:
            sound = AudioSegment.from_wav(audio_file)
            sound.export("audio.wav", format="wav")
            st.success("File Saved")

        run_ser_btn = st.button("Run SER")
        if run_ser_btn:
            exec(open("SpeechRecognition.py").read())
            emotion_speech_final = open('FinalSpeechEmotion.txt', 'r').read()
            st.subheader("Predicted Emotion :")
            st.subheader(emotion_speech_final)
            image = Image.open('speechemotion.jpg')
            st.image(image)
            self.display_activities(emotion_speech_final)

if __name__ == "__main__":
    ars = ActivityRecommendationSystem("Activities.json")
    st.title("Activity Recommendation System")
    st.write("This is an Activity Recommendation System based on Convolution Neural Networks that suggests appropriate mood enhancing activities to counter anxiety and depression by examining users' facial expressions and voice intensities.")
    ars.fer_upload_choice()
    ars.ser_upload_choice()
