import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parameters
DATA_PATH = r"D:\ISL\GESTURES"
SEQUENCE_LENGTH = 30  # Assuming 30 frames per gesture (30fps for ~1 second)
MAX_HANDS = 2  # Maximum number of hands to detect
NUM_LANDMARKS = 21  # Number of landmarks per hand
NUM_FEATURES = 3  # x, y, z coordinates per landmark
EPOCHS = 50
BATCH_SIZE = 16

def extract_keypoints(frame, hands):
    """Extract hand keypoints from a frame using MediaPipe."""
    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and detect hands
    results = hands.process(image)
    
    # Initialize empty array to store keypoints
    keypoints = np.zeros(MAX_HANDS * NUM_LANDMARKS * NUM_FEATURES)
    
    if results.multi_hand_landmarks:
        # Loop through detected hands
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= MAX_HANDS:
                break
                
            # Extract landmarks for each hand
            for landmark_idx, landmark in enumerate(hand_landmarks.landmark):
                # Calculate the base index for this landmark in the flat array
                base_idx = hand_idx * NUM_LANDMARKS * NUM_FEATURES + landmark_idx * NUM_FEATURES
                
                # Store x, y, z coordinates
                keypoints[base_idx] = landmark.x
                keypoints[base_idx + 1] = landmark.y
                keypoints[base_idx + 2] = landmark.z
                
    return keypoints

def process_video(video_path, hands):
    """Process a video file and return sequence of keypoints."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Read all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    # Calculate step size to sample SEQUENCE_LENGTH frames
    total_frames = len(frames)
    if total_frames <= SEQUENCE_LENGTH:
        # If the video has fewer frames than needed, duplicate the last frame
        frame_indices = list(range(total_frames))
        frame_indices += [total_frames-1] * (SEQUENCE_LENGTH - total_frames)
    else:
        # Sample frames evenly throughout the video
        step = total_frames / SEQUENCE_LENGTH
        frame_indices = [int(i * step) for i in range(SEQUENCE_LENGTH)]
    
    # Extract keypoints from selected frames
    keypoints_sequence = []
    for idx in frame_indices:
        if idx < len(frames):
            keypoints = extract_keypoints(frames[idx], hands)
            keypoints_sequence.append(keypoints)
    
    return np.array(keypoints_sequence)

def load_dataset():
    """Load dataset and extract features."""
    X = []
    y = []
    
    # Initialize MediaPipe hands with appropriate settings
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        # Process each gesture class
        for gesture_idx, gesture_folder in enumerate(sorted(os.listdir(DATA_PATH))):
            gesture_path = os.path.join(DATA_PATH, gesture_folder)
            
            # Skip if not a directory
            if not os.path.isdir(gesture_path):
                continue
                
            gesture_name = gesture_folder.split('. ')[-1] if '. ' in gesture_folder else gesture_folder
            print(f"Processing gesture: {gesture_name}")
            
            # Process each video in the gesture folder
            for video_file in os.listdir(gesture_path):
                # Skip non-video files (assuming videos are .mp4, .avi, etc.)
                if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue
                
                video_path = os.path.join(gesture_path, video_file)
                print(f"  Processing video: {video_file}")
                
                try:
                    # Process video and extract keypoints sequence
                    keypoints_sequence = process_video(video_path, hands)
                    
                    # Add to dataset
                    X.append(keypoints_sequence)
                    y.append(gesture_name)
                except Exception as e:
                    print(f"Error processing video {video_file}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def create_model(input_shape, num_classes):
    """Create LSTM model for gesture recognition."""
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    # Load dataset
    print("Loading dataset...")
    X, y = load_dataset()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    print(f"Input shape: {X_train.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_encoder.classes_)
    model = create_model(input_shape, num_classes)
    
    # Create callbacks
    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Save model and label encoder
    model.save('isl_gesture_model.keras')
    np.save('label_encoder_classes.npy', label_encoder.classes_)
    
    print("Model saved as 'isl_gesture_model.keras'")
    print("Label encoder classes saved as 'label_encoder_classes.npy'")
    
    return model, label_encoder, history

def predict_gesture(model, label_encoder, video_path):
    """Predict gesture for a new video."""
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        # Process video and extract keypoints sequence
        keypoints_sequence = process_video(video_path, hands)
        
        # Reshape for prediction
        X = np.expand_dims(keypoints_sequence, axis=0)
        
        # Predict
        prediction = model.predict(X)
        gesture_idx = np.argmax(prediction[0])
        gesture_name = label_encoder.classes_[gesture_idx]
        confidence = prediction[0][gesture_idx]
        
        return gesture_name, confidence

if __name__ == "__main__":
    # Train model
    model, label_encoder, history = train_model()
    
    # Example: Predict gesture for a new video
    # Uncomment and modify path as needed
    # test_video_path = "path/to/test/video.mp4"
    # gesture_name, confidence = predict_gesture(model, label_encoder, test_video_path)
    # print(f"Predicted gesture: {gesture_name} (confidence: {confidence:.4f})")