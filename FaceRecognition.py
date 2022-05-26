# Face Recognition:
import json
import random
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import model_from_json
model_json_file = 'C:/Users/ghare/Desktop/BE Project Codes G86/model.json'
model_weights_file = 'C:/Users/ghare/Desktop/BE Project Codes G86/model_weights.h5'
json_file_path = "C:/Users/ghare/Desktop/BE Project Codes G86/Activities.json"

with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_file)
    
face_cascade = cv2.CascadeClassifier('C:/Users/ghare/Desktop/BE Project Codes G86/haarcascade_frontalface_default.xml')

image = cv2.imread('C:/Users/ghare/Desktop/BE Project Codes G86/image.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
faces = face_cascade.detectMultiScale(gray)

for (x,y,w,h) in faces:
  fc = gray[y:y+h, x:x+w]
 
  roi = cv2.resize(fc, (48,48))
  pred = loaded_model.predict(roi[np.newaxis, :, :, np.newaxis])
  text_idx=np.argmax(pred)
  text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
  
  if text_idx == 0:
    text= text_list[0]
  elif text_idx == 1:
    text= text_list[1]
  elif text_idx == 2:
    text= text_list[2]
  elif text_idx == 3:
    text= text_list[3]
  elif text_idx == 4:
    text= text_list[4]
  elif text_idx == 5:
    text= text_list[5]
  elif text_idx == 6:
    text= text_list[6]
  
  cv2.putText(gray, text, (x, y-5),
  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
  img2 = cv2.rectangle(gray, (x,y), (x+w, y+h), (0,0,255), 2)           
    
# cv2_imshow(img2)
print(text)
# cap.release()
cv2.destroyAllWindows()
file1 = open("C:/Users/ghare/Desktop/BE Project Codes G86/FinalFaceEmotion.txt", "w")
file1.write(text)
file1.close()
cv2.imwrite("C:/Users/ghare/Desktop/BE Project Codes G86/faceemotion.jpg", img2)              