Task 04: Real-Time Hand Gesture Recognition

#Project Overview
This project is a real-time computer vision application that identifies various hand gestures through a webcam feed. By leveraging Deep Learning (CNN) and OpenCV, the system can distinguish between different hand shapes and display the results instantly on a web-based dashboard. This was developed as part of my Machine Learning Internship at Prodigy InfoTech.

#Technical Stack
Deep Learning Framework: TensorFlow & Keras

Computer Vision: OpenCV (cv2)

Web Interface: Streamlit

Programming Language: Python

Numerical Processing: NumPy

#Features
Live Webcam Integration: Processes video frames at high FPS for smooth recognition.

Advanced Image Preprocessing: Includes Grayscale conversion and Gaussian Blurring to improve model accuracy in different lighting conditions.

Interactive UI: A clean Streamlit interface with a "Start Webcam" toggle and real-time confidence scoring.

Gesture Categories: Capable of recognizing Palm, Fist, Thumbs Up, and more from the LeapGestRecog dataset.

#How to Setup & Run

*Clone the repository:
git clone https://github.com/yourusername/Prodigy_ML_04.git
cd Prodigy_ML_04

*Install dependencies:
pip install tensorflow streamlit opencv-python numpy

*Launch the application:
streamlit run gesture_app.py

#Model Logic
The core of this project is a Convolutional Neural Network (CNN). The model was trained to recognize spatial patterns in hand images. During execution, the app defines a Region of Interest (ROI)—the blue box on your screen—and passes that specific data to the model for a lightning-fast prediction.
