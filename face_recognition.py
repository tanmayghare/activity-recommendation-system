import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

class FaceRecognition:
    def __init__(self, model_json_file, model_weights_file, face_cascade_file):
        # Initialize the FaceRecognition class with model and face cascade files
        self.model = self.load_model(model_json_file, model_weights_file)
        self.face_cascade = cv2.CascadeClassifier(face_cascade_file)
        self.text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def load_model(self, model_json_file, model_weights_file):
        # Load the model from JSON and weights files
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(model_weights_file)
        return model

    def predict_emotion(self, image_path):
        # Predict the emotion from the given image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            fc = gray[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = self.model.predict(roi[np.newaxis, :, :, np.newaxis])
            text_idx = np.argmax(pred)
            text = self.text_list[text_idx]
            cv2.putText(gray, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
            img2 = cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.imwrite("faceemotion.jpg", img2)
            with open("FinalFaceEmotion.txt", "w") as file:
                file.write(text)
            return text

if __name__ == "__main__":
    face_recognition = FaceRecognition('model.json', 'model_weights.h5', 'haarcascade_frontalface_default.xml')
    emotion = face_recognition.predict_emotion('image.jpg')
    print(emotion)
    cv2.destroyAllWindows()
