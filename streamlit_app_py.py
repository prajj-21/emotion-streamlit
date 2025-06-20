# -*- coding: utf-8 -*-
"""streamlit_app.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TA43GngphhPutKE1EXiL7lwq2nMYc__d
"""

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile

# Load the trained model
model = load_model('emotion_model.h5')  # Change to your actual model file name
IMG_SIZE = 48

# Class label map
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

def detect_spoof(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return "Real" if variance > 100 else "Spoof"

def detect_emotion_and_spoof(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face_norm = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(np.expand_dims(face_norm, -1), 0)

        prediction = model.predict(face_input)
        emotion_idx = np.argmax(prediction)
        emotion_label = emotion_dict[emotion_idx]

        spoof_label = detect_spoof(image[y:y+h, x:x+w])
        label = f"{emotion_label} | {spoof_label}"

        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    return image

# Streamlit UI
st.title("Emotion Detection with Spoofing Check")
st.write("Upload an image or click a photo using webcam")

option = st.radio("Choose input method:", ["Upload Image", "Take Photo"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        result = detect_emotion_and_spoof(image)
        st.image(result, channels="BGR", caption="Result")

elif option == "Take Photo":
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture)
        image = np.array(image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = detect_emotion_and_spoof(image)
        st.image(result, channels="BGR", caption="Result")
