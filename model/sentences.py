import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import os

# Load the trained model
model = load_model('isl_to_text_cnn_model.h5')

# Load MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Get gesture labels from the folder names in your data directory
GESTURES = sorted(os.listdir('gesture_data'))
NUM_FRAMES = 30
NUM_LANDMARKS = 21 * 3
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for accepting a prediction

# Start video capture
cap = cv2.VideoCapture(0)

# Buffers for smoothing and building sentences
sequence = []
buffer = deque(maxlen=5)  # To smooth predictions
sentence = ""  # To hold the output sentence
last_word = ""  # To prevent immediate repetition of words

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                sequence.append(landmarks)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Predict gestures when NUM_FRAMES are collected
            if len(sequence) == NUM_FRAMES:
                sequence_np = np.array(sequence).reshape(1, NUM_FRAMES, NUM_LANDMARKS)
                prediction = model.predict(sequence_np)[0]
                max_index = np.argmax(prediction)
                confidence = prediction[max_index]

                if confidence > CONFIDENCE_THRESHOLD:
                    buffer.append(GESTURES[max_index])

                # Clear sequence for the next prediction
                sequence = []

        # Smooth predictions using a majority vote in the buffer
        if len(buffer) > 0:
            current_word = max(set(buffer), key=buffer.count)
            if current_word != last_word:
                sentence += current_word + " "  # Append the new word with a space
                last_word = current_word

        # Display the dynamically forming sentence
        cv2.putText(frame, sentence.strip(), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the output frame
        cv2.imshow('Real-Time Gesture to Sentence', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()
