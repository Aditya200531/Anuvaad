import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands

# Parameters
DATA_PATH = r"D:\ISL\GESTURES"
SEQUENCE_LENGTH = 30  # Fixed number of frames for each video
FRAME_HEIGHT = 128  # Resize frames to this height
FRAME_WIDTH = 128   # Resize frames to this width
MAX_HANDS = 2       # Maximum number of hands to detect in frame

def create_heatmap_from_landmarks(landmarks, frame_shape):
    """
    Convert hand landmarks to heatmap representation.
    Creates one channel for each hand detected (up to MAX_HANDS).
    """
    heatmaps = []
    
    height, width = frame_shape[:2]
    
    for hand_idx in range(MAX_HANDS):
        # Create empty heatmap for this hand
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Get landmarks for this hand if available
        hand_landmarks = landmarks[hand_idx] if hand_idx < len(landmarks) else None
        
        if hand_landmarks:
            # Draw landmarks as Gaussian blobs
            for landmark in hand_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                x, y = int(landmark.x * width), int(landmark.y * height)
                
                # Skip if outside frame
                if 0 <= x < width and 0 <= y < height:
                    # Create a Gaussian blob around the landmark
                    size = 5  # Size of the Gaussian kernel
                    for dx in range(-size, size + 1):
                        for dy in range(-size, size + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                # Gaussian falloff based on distance
                                dist = (dx**2 + dy**2) / (2 * (size/3)**2)
                                intensity = np.exp(-dist)
                                heatmap[ny, nx] = max(heatmap[ny, nx], intensity)
        
        heatmaps.append(heatmap)
    
    # Combine heatmaps into a multi-channel image
    multi_channel = np.stack(heatmaps, axis=-1)
    return multi_channel

def process_video(video_path):
    """Process a video file and return a tensor suitable for 3D CNN."""
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
    
    # Process selected frames with MediaPipe and convert to tensors
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        processed_frames = []
        
        for idx in frame_indices:
            if idx < len(frames):
                # Resize frame
                frame = frames[idx]
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                
                # Process with MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    # Create heatmap representation of hand landmarks
                    heatmap = create_heatmap_from_landmarks(
                        results.multi_hand_landmarks, 
                        (FRAME_HEIGHT, FRAME_WIDTH)
                    )
                else:
                    # No hands detected
                    heatmap = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, MAX_HANDS), dtype=np.float32)
                
                # Create RGB + heatmap representation
                # 3 channels for RGB + MAX_HANDS channels for heatmaps
                frame_normalized = frame.astype(np.float32) / 255.0
                combined = np.concatenate([frame_normalized, heatmap], axis=-1)
                
                processed_frames.append(combined)
    
    # Convert to numpy array with shape [SEQUENCE_LENGTH, HEIGHT, WIDTH, CHANNELS]
    tensor = np.array(processed_frames)
    return tensor

def load_dataset():
    """Process videos and prepare dataset for 3D CNN."""
    X = []
    y = []
    
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
            # Skip non-video files
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue
            
            video_path = os.path.join(gesture_path, video_file)
            print(f"  Processing video: {video_file}")
            
            try:
                # Process video and extract tensor
                video_tensor = process_video(video_path)
                
                # Add to dataset
                X.append(video_tensor)
                y.append(gesture_name)
            except Exception as e:
                print(f"Error processing video {video_file}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def process_and_save_dataset():
    """Process all videos and save the dataset to disk."""
    print("Processing dataset for 3D CNN...")
    X, y = load_dataset()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Print dataset info
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Save processed data
    np.save('3dcnn_features.npy', X)
    np.save('3dcnn_labels.npy', y_encoded)
    np.save('3dcnn_label_classes.npy', label_encoder.classes_)
    
    print("Dataset processed and saved to disk.")
    
    return X, y_encoded, label_encoder

if __name__ == "__main__":
    # Process videos and save features for 3D CNN
    X, y_encoded, label_encoder = process_and_save_dataset()