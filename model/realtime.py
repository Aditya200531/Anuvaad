import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model('gesture_recognition_best_model.h5')

# Load MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Get gesture labels from the folder names in your data directory
GESTURES = sorted(os.listdir('gesture_data'))
NUM_FRAMES = 30
NUM_LANDMARKS = 21 * 3

# Start video capture
cap = cv2.VideoCapture(0)

# List to store detected gestures and recent predictions for smoothing
detected_gestures = []
recent_predictions = []
confidence_threshold = 0.7  # Confidence threshold

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    sequence = []
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

                # Draw landmarks on the image
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Predict gestures if 30 frames (sequence) are collected
            if len(sequence) == NUM_FRAMES:
                sequence_np = np.array(sequence).reshape(1, NUM_FRAMES, NUM_LANDMARKS)  # Ensure correct shape
                prediction = model.predict(sequence_np)[0]
                max_index = np.argmax(prediction)
                confidence = prediction[max_index]
                
                if confidence > confidence_threshold:
                    gesture = GESTURES[max_index]
                    detected_gestures.append(gesture)
                    recent_predictions.append((gesture, confidence))
                    if len(detected_gestures) > 10:
                        detected_gestures.pop(0)
                        recent_predictions.pop(0)
                
                # Clear sequence for next prediction cycle
                sequence = []

        # Display the detected gestures as a sentence with confidence
        sentence = ' '.join([f"{gesture} ({confidence:.2f})" for gesture, confidence in recent_predictions])

        # Split the sentence into multiple lines if it's too long for one line
        y0, dy = 50, 30  # Starting Y position and line height
        max_line_length = 30  # Maximum number of characters per line
        wrapped_sentence = [sentence[i:i + max_line_length] for i in range(0, len(sentence), max_line_length)]

        # Display each line
        for i, line in enumerate(wrapped_sentence):
            cv2.putText(frame, line, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the output frame
        cv2.imshow('Real-Time Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()
