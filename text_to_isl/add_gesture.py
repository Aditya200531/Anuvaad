import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Directory setup for storing data
DATA_DIR = "gesture_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Function for outlier detection
def is_outlier(landmarks):
    coords = np.array(landmarks).flatten()
    z_scores = (coords - np.mean(coords)) / np.std(coords)
    return np.any(np.abs(z_scores) > 2.5)  # Flag as outlier if z-score exceeds threshold

# Function to collect data for a gesture
def collect_data(gesture_label):
    cap = cv2.VideoCapture(0)
    collected_data = []
    print(f"Starting data collection for gesture '{gesture_label}'. Press 'q' to stop recording.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract coordinates
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                if not is_outlier(landmarks):  # Filter outliers
                    collected_data.append([coord for lm in landmarks for coord in lm])

                # Draw landmarks on the screen
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Gesture Recorder', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected_data:
        # Save data
        df = pd.DataFrame(collected_data)
        df.to_csv(os.path.join(DATA_DIR, f'gesture_{gesture_label}.csv'), index=False)
        print(f"Data for gesture '{gesture_label}' saved successfully!")
        messagebox.showinfo("Data Collection", f"Gesture '{gesture_label}' data collected and saved.")
    else:
        print(f"No data collected for gesture '{gesture_label}'.")
        messagebox.showwarning("Data Collection", f"No data recorded for gesture '{gesture_label}'.")

# GUI for data collection
def start_data_collection():
    root = tk.Tk()
    root.title("Hand Gesture Data Collection")

    def on_collect(gesture_label):
        messagebox.showinfo("Instructions", f"Recording data for Gesture {gesture_label}. Press 'q' to stop.")
        collect_data(gesture_label)

    for i in range(1, 6):
        button = tk.Button(root, text=f"Record Gesture {i}", command=lambda i=i: on_collect(i))
        button.pack(pady=5)

    root.mainloop()

start_data_collection()

# Preprocess the data
def load_and_preprocess_data():
    X, y = [], []
    for i in range(1, 6):
        file_path = os.path.join(DATA_DIR, f'gesture_{i}.csv')
        data = pd.read_csv(file_path).values
        X.append(data)
        y.append(np.full(len(data), i - 1))  # Label as 0-indexed class

    X = np.vstack(X)
    y = np.hstack(y)

    # Normalize the data
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    return X, y

X, y = load_and_preprocess_data()

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build MobileNetV2 model
def create_model(input_shape):
    base_model = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
    base_model.trainable = True

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(5, activation='softmax')
    ])
    return model

input_shape = (21, 3)  # Number of landmarks, each with (x, y, z)
model = create_model((input_shape[0], input_shape[1], 1))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('gesture_model_2.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model training complete and saved as gesture_model.tflite.")
messagebox.showinfo("Model Training", "Model training complete. Saved as gesture_model.tflite.")
