import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import argparse

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parameters
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
        visualization_frames = []
        
        for idx in frame_indices:
            if idx < len(frames):
                # Resize frame
                frame = frames[idx]
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                
                # Make a copy for visualization
                vis_frame = frame.copy()
                
                # Process with MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                # Draw landmarks on visualization frame
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            vis_frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS)
                    
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
                visualization_frames.append(vis_frame)
    
    # Convert to numpy array with shape [SEQUENCE_LENGTH, HEIGHT, WIDTH, CHANNELS]
    tensor = np.array(processed_frames)
    return tensor, visualization_frames

def predict_gesture(model, label_encoder, video_path, visualize=False):
    """Predict gesture for a new video."""
    # Process video and extract tensor
    video_tensor, vis_frames = process_video(video_path)
    
    # Reshape for prediction (add batch dimension)
    X = np.expand_dims(video_tensor, axis=0)
    
    # Predict
    prediction = model.predict(X)
    gesture_idx = np.argmax(prediction[0])
    gesture_name = label_encoder.classes_[gesture_idx]
    confidence = prediction[0][gesture_idx]
    
    # Get top 3 predictions
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    top_gestures = [(label_encoder.classes_[idx], prediction[0][idx]) for idx in top_indices]
    
    # Visualize if requested
    if visualize:
        output_path = os.path.join(os.path.dirname(video_path), 
                                   f"prediction_{os.path.basename(video_path)}")
        visualize_prediction(vis_frames, gesture_name, confidence, top_gestures, output_path)
    
    return gesture_name, confidence, top_gestures

def visualize_prediction(frames, gesture_name, confidence, top_gestures, output_path):
    """Create visualization video with prediction results."""
    if not frames:
        print("No frames to visualize.")
        return
        
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    # Add prediction overlay to each frame
    for frame in frames:
        # Add top predictions text
        y_pos = 30
        cv2.putText(frame, f"Prediction: {gesture_name} ({confidence:.2f})", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add top 3 predictions
        for i, (gesture, conf) in enumerate(top_gestures):
            y_pos += 20
            cv2.putText(frame, f"{i+1}. {gesture}: {conf:.2f}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # Write frame to video
        out.write(frame)
    
    out.release()
    print(f"Visualization saved to {output_path}")

def create_prediction_visualization(model, label_encoder, video_path):
    """Create a detailed visualization of the prediction process."""
    # Process video and get prediction
    video_tensor, frames = process_video(video_path)
    
    # Reshape for prediction
    X = np.expand_dims(video_tensor, axis=0)
    
    # Predict
    prediction = model.predict(X)
    gesture_idx = np.argmax(prediction[0])
    gesture_name = label_encoder.classes_[gesture_idx]
    confidence = prediction[0][gesture_idx]
    
    # Get all gesture probabilities
    all_probs = [(label_encoder.classes_[i], prediction[0][i]) 
                 for i in range(len(label_encoder.classes_))]
    all_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot sample frames
    num_sample_frames = min(5, len(frames))
    sample_indices = np.linspace(0, len(frames)-1, num_sample_frames, dtype=int)
    
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, num_sample_frames, i+1)
        plt.imshow(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {idx}")
        plt.axis('off')
    
    # Plot prediction probabilities
    plt.subplot(2, 1, 2)
    gestures = [g for g, _ in all_probs]
    probs = [p for _, p in all_probs]
    
    bars = plt.bar(gestures, probs)
    plt.xlabel('Gesture')
    plt.ylabel('Probability')
    plt.title(f'Prediction: {gesture_name} (Confidence: {confidence:.2f})')
    plt.xticks(rotation=45, ha='right')
    
    # Highlight the predicted gesture
    bars[gesture_idx].set_color('green')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(video_path), 
                              f"analysis_{os.path.basename(video_path)}.png")
    plt.savefig(output_path)
    print(f"Analysis visualization saved to {output_path}")
    
    return gesture_name, confidence

def main():
    parser = argparse.ArgumentParser(description='Predict ISL gesture from video using 3D CNN')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--visualize', action='store_true', help='Create visualization video')
    parser.add_argument('--analysis', action='store_true', help='Create detailed analysis visualization')
    
    args = parser.parse_args()
    
    # Load model and label encoder
    try:
        model = tf.keras.models.load_model('isl_3dcnn_model.keras')
        label_encoder_classes = np.load('3dcnn_label_classes.npy', allow_pickle=True)
        
        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_encoder_classes
        
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Model or label encoder not found. Please train the model first.")
        return
    
    # Predict gesture
    if args.analysis:
        gesture_name, confidence = create_prediction_visualization(
            model, label_encoder, args.video_path)
            
        print(f"\nPredicted gesture: {gesture_name}")
        print(f"Confidence: {confidence:.4f}")
    else:
        gesture_name, confidence, top_gestures = predict_gesture(
            model, label_encoder, args.video_path, args.visualize)
            
        print(f"\nPredicted gesture: {gesture_name}")
        print(f"Confidence: {confidence:.4f}")
        
        print("\nTop 3 predictions:")
        for i, (gesture, conf) in enumerate(top_gestures):
            print(f"{i+1}. {gesture}: {conf:.4f}")

if __name__ == "__main__":
    main()