import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('gesture_model.h5')
LABELS = ['Palm', 'L', 'Fist', 'Fist_Moved', 'Thumb', 'Index', 'OK', 'Palm_Moved', 'C', 'Down']

st.title("Robust CNN Gesture Recognizer")
st.markdown("**Instructions:** Place your hand inside the blue box.")
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

if run:
    cam = cv2.VideoCapture(0)
    while run:
        ret, frame = cam.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        h, w, c = frame.shape
        x1, y1 = int(w/2) - 150, int(h/2) - 150
        x2, y2 = int(w/2) + 150, int(h/2) + 150
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        
        resized = cv2.resize(blur, (64, 64))
        normalized = resized.reshape(1, 64, 64, 1) / 255.0
        prediction = model.predict(normalized, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        if confidence > 0.80:
            label = f"{LABELS[class_idx]} ({confidence*100:.0f}%)"
            color = (0, 255, 0) 
        else:
            label = "Waiting for hand..."
            color = (0, 0, 255) 
            
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    cam.release()