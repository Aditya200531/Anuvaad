import os
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parameters
SEQUENCE_LENGTH = 30  # Assuming 30 frames per gesture (30fps for ~1 second)
MAX_HANDS = 2  # Maximum number of hands to detect
NUM_LANDMARKS = 21  # Number of landmarks per hand
NUM_FEATURES = 3  # x, y, z coordinates per landmark

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

def load_dataset(data_path):
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
        for gesture_idx, gesture_folder in enumerate(sorted(os.listdir(data_path))):
            gesture_path = os.path.join(data_path, gesture_folder)
            
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