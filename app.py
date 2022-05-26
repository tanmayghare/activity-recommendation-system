import io
import cv2
import json
import numpy as np
import os
import streamlit as st
import webbrowser
from PIL import Image
from pydub import AudioSegment
import random
from scipy.io.wavfile import read, write

json_file_path = "C:/Users/ghare/Desktop/BE Project Codes G86/Activities.json"

st.title("Activity Recommendation System")
st.write("This is an Activity Recommendation System based on Convolution Neural Networks that suggests appropriate mood enhancing activities to counter anxiety and depression by examining users' facial expressions and voice intensities.")

def activities(index, emotion):
  global sub_key, sub_value, tempurl
  with open(json_file_path, 'r') as j:
      contents = json.loads(j.read())
      for keys in contents["activities"][index][emotion][0].values():
          for sub_key, sub_value in keys.items():

              idx = random.randrange(0, len(sub_value))
              url = '[' +sub_key +']' +'(' +sub_value[idx]+')'
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
                          ''',
                          unsafe_allow_html=True
                      )


def FERUploadchoice():
# Face Expression Recognition:
  st.header("Face Expression Recognition:")

  ferChoice = st.radio("Please choose one option :", ("Upload Image", "Capture Image"))

  if ferChoice == "Upload Image":
    image_file = st.file_uploader("Upload Image :", type=["png","jpg","jpeg"])
    if image_file is not None:
      st.image(image_file, width=250)
    #Saving upload
      with open(os.path.join("C:/Users/ghare/Desktop/BE Project Codes G86", 'image.jpg'),"wb") as f:
        f.write((image_file).getbuffer())

      st.success("File Saved")

  if ferChoice == "Capture Image":
    captured_image = st.camera_input("Take a picture :")

    if captured_image is not None:
      st.success("Image Captured")
      # To read image file buffer with OpenCV:
      bytes_data = captured_image.getvalue()
      cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
      with open(os.path.join("C:/Users/ghare/Desktop/BE Project Codes G86/", 'image.jpg'), "wb") as f:
        f.write((captured_image).getbuffer())

  runFerBtn = st.button("Run FER")

  if runFerBtn:
    try:
      exec(open("C:/Users/ghare/Desktop/BE Project Codes G86/FaceRecognition.py").read())
    except:
      st.write("Please re-capture the image")
        
    emotionFaceFinal = open('C:/Users/ghare/Desktop/BE Project Codes G86/FinalFaceEmotion.txt', 'r').read()
    st.subheader("Predicted Emotion: " + emotionFaceFinal)
    # st.subheader(emotionFaceFinal)
    image = Image.open('C:/Users/ghare/Desktop/BE Project Codes G86/faceemotion.jpg')
    st.image(image)

    with open(json_file_path, 'r') as j:
      contents = json.loads(j.read())

      if emotionFaceFinal == "Happy":        
        st.subheader("Voila!, maybe let's boost your happiness a bit more. Suggesting you some activities you might find useful:")
        activities(0, "happy")

      if emotionFaceFinal == "Sad":
        st.subheader("Oh dear, don't worry we understand you. Suggesting you some activities you might find useful:")
        activities(1, "sad")

      if emotionFaceFinal == "Angry":
        st.subheader("Calm down!, We'll help you with doing so ;). Suggesting you some activities you might find useful:")
        activities(2, "angry")        

      if emotionFaceFinal == "Neutral":
        st.subheader("I see, anyways. Suggesting you some activities you might find useful:")
        activities(3, "neutral")            

      if emotionFaceFinal == "Surprise":
          st.subheader("Heard things you shouldn't have, totally fine. Suggesting you some activities you might find useful:")
          activities(4, "surprised")

      if emotionFaceFinal == "Fearful":
          st.subheader("Why fear when we are here. Suggesting you some activities you might find useful:")
          activities(5, "fearful")

      if emotionFaceFinal == "Disgust":
          st.subheader("Chillofy!. Suggesting you some activities you might find useful:")
          activities(6, "disgust")

      if emotionFaceFinal == "Calm":
          st.subheader("Yes you should be, always!. Suggesting you some activities you might find useful:")
          activities(7, "calm")                                                      

def SERUploadchoice():
# Speech Emotion Recognition:
  st.header("Speech Emotion Recognition:")

  # serChoice = st.radio("Please choose one option :", ("Upload Audio", "Record Audio"))

  # if serChoice == "Upload Audio":
  audio_file = st.file_uploader("Upload Audio Recording :", type=["wav"])
  if audio_file is not None:
    sound = AudioSegment.from_wav(audio_file)
    sound.export("C:/Users/ghare/Desktop/BE Project Codes G86/audio.wav", format="wav")

    st.success("File Saved")

  # if serChoice == "Record Audio":

  #   import sounddevice as sd
  #   from scipy.io.wavfile import write
  #   import wavio as wv

  #   freq = 48000
  #   duration = 3
  #   st.write("Recording Voice... :")
  #   recording = sd.rec(int(duration * freq), 
  #                     samplerate=freq, channels=2)
      
  #   sd.wait()
    
  #   # write("audio0.wav", freq, recording)
  #   # Convert the NumPy array to audio file
  #   wv.write("audio.wav", recording, freq, sampwidth=2)

  #   st.success("Voice Recorded Successfully")

  runSerBtn = st.button("Run SER")  

  if runSerBtn:
    exec(open("C:/Users/ghare/Desktop/BE Project Codes G86/SpeechRecognition.py").read())
    emotionSpeechFinal = open('C:/Users/ghare/Desktop/BE Project Codes G86/FinalSpeechEmotion.txt', 'r').read()
    st.subheader("Predicted Emotion :")
    st.subheader(emotionSpeechFinal)
    image = Image.open('C:/Users/ghare/Desktop/BE Project Codes G86/speechemotion.jpg')
    st.image(image)

    with open(json_file_path, 'r') as j:
      contents = json.loads(j.read())

      if emotionSpeechFinal == "Happy":        
        st.subheader("Voila!, maybe let's boost your happiness a bit more. Suggesting you some activities you might find useful:")
        activities(0, "happy")

      if emotionSpeechFinal == "Sad":
        st.subheader("Oh dear, don't worry we understand you. Suggesting you some activities you might find useful:")
        activities(1, "sad")             

      if emotionSpeechFinal == "Angry":
        st.subheader("Calm down!, We'll help you with doing so ;). Suggesting you some activities you might find useful:")
        activities(2, "angry")

      if emotionSpeechFinal == "Neutral":
        st.subheader("I see, anyways. Suggesting you some activities you might find useful:")
        activities(3, "neutral")

      if emotionSpeechFinal == "Surprise":
          st.subheader("Heard things you shouldn't have, totally fine. Suggesting you some activities you might find useful:")
          activities(4, "surprised")

      if emotionSpeechFinal == "Fearful":
          st.subheader("Why fear when we are here. Suggesting you some activities you might find useful:")
          activities(5, "fearful")

      if emotionSpeechFinal == "Disgust":
          st.subheader("Chillofy!. Suggesting you some activities you might find useful:")
          activities(6, "disgust")

      if emotionSpeechFinal == "Calm":
          st.subheader("Yes you should be, always!. Suggesting you some activities you might find useful:")
          activities(7, "calm")

FERUploadchoice()
SERUploadchoice()