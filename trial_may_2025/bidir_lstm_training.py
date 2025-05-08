import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Masking, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Bidirectional, Concatenate, GlobalAveragePooling1D, Add, MultiHeadAttention
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parameters
DATA_PATH = r"C:\Users\kusha\OneDrive\Documents\Programming\Hackathons\smart_india_hackathon\trial_may_2025\Gestures"
MAX_HANDS = 2          # maximum number of hands to detect
NUM_LANDMARKS = 21     # landmarks per hand
NUM_FEATURES = 3       # x, y, z coords per landmark
EPOCHS = 100           # increased from 50
BATCH_SIZE = 32        # increased from 16
MODEL_DIR = r"lstm_models"  # directory to save models
AUGMENT_DATA = True     # whether to use data augmentation
CROSS_VAL_FOLDS = 5     # number of folds for cross-validation

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def extract_enhanced_keypoints(frame, hands):
    """Extract hand keypoints with advanced features from a frame using MediaPipe."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    # Initialize output array for all hands
    keypoints = np.zeros(MAX_HANDS * NUM_LANDMARKS * NUM_FEATURES)
    
    if results.multi_hand_landmarks:
        for h_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if h_idx >= MAX_HANDS:
                break
                
            # Extract basic coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))
            landmarks = np.array(landmarks)
            
            # Find reference points (wrist)
            wrist = landmarks[0]
            
            # Normalize coordinates relative to wrist
            landmarks_normalized = landmarks - wrist
            
            # Add scale normalization
            # Calculate bounding box dimensions
            min_vals = np.min(landmarks[:, :2], axis=0)
            max_vals = np.max(landmarks[:, :2], axis=0)
            bbox_size = np.max(max_vals - min_vals)
            if bbox_size > 0:  # Avoid division by zero
                landmarks_normalized[:, :2] = landmarks_normalized[:, :2] / bbox_size
            
            # Store normalized landmarks in the output array
            for l_idx, landmark_norm in enumerate(landmarks_normalized):
                base = h_idx * NUM_LANDMARKS * NUM_FEATURES + l_idx * NUM_FEATURES
                keypoints[base:base+3] = landmark_norm
                
    return keypoints

def apply_smoothing(sequence, window_size=3):
    """Apply temporal smoothing to reduce noise in the keypoint sequence"""
    if sequence.shape[0] <= window_size:
        return sequence
    
    smoothed = np.copy(sequence)
    for i in range(window_size, sequence.shape[0] - window_size):
        smoothed[i] = np.mean(sequence[i-window_size:i+window_size+1], axis=0)
    
    return smoothed

def process_video_enhanced(video_path, hands):
    """Process video with enhanced features and smoothing"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Extract keypoints for each frame
    seq = [extract_enhanced_keypoints(f, hands) for f in frames]
    if not seq:
        return np.zeros((0, MAX_HANDS * NUM_LANDMARKS * NUM_FEATURES))
    
    # Convert to numpy array and apply smoothing
    seq = np.array(seq)
    seq = apply_smoothing(seq)
    
    return seq

def augment_keypoints(keypoints, augmentation_factor=3):
    """Apply augmentation to hand keypoint data"""
    if keypoints.shape[0] == 0:
        return [keypoints]
        
    augmented = [keypoints]
    
    # Add slight random noise to coordinates
    noise_seq = keypoints + np.random.normal(0, 0.005, keypoints.shape)
    augmented.append(noise_seq)
    
    # Scale keypoints slightly
    for scale in [0.95, 1.05]:
        scaled = keypoints.copy()
        # Only scale x,y (not z)
        for i in range(0, scaled.shape[1], 3):
            scaled[:, i:i+2] *= scale
        augmented.append(scaled)
    
    # Add slight time warping (speed variation)
    if keypoints.shape[0] > 15:
        # Create faster version (90% speed)
        faster_len = int(keypoints.shape[0] * 0.9)
        faster = np.zeros((faster_len, keypoints.shape[1]))
        for i in range(faster_len):
            orig_i = min(int(i * (keypoints.shape[0] / faster_len)), keypoints.shape[0] - 1)
            faster[i] = keypoints[orig_i]
        augmented.append(faster)
        
        # Create slower version (110% speed)
        slower_len = int(keypoints.shape[0] * 1.1)
        slower = np.zeros((slower_len, keypoints.shape[1]))
        for i in range(slower_len):
            orig_i = min(int(i * (keypoints.shape[0] / slower_len)), keypoints.shape[0] - 1)
            slower[i] = keypoints[orig_i]
        augmented.append(slower)
    
    # Horizontal flip (for gestures where left/right doesn't matter)
    flipped = keypoints.copy()
    for i in range(0, flipped.shape[1], 3):
        flipped[:, i] = -flipped[:, i]  # Flip x coordinates
    augmented.append(flipped)
    
    # Return subset based on augmentation factor
    return augmented[:augmentation_factor]

def load_and_process_dataset(augment=True, augmentation_factor=3):
    """Load dataset with optional augmentation and enhanced preprocessing"""
    seqs, labels = [], []
    class_counts = {}
    
    with mp_hands.Hands(static_image_mode=False,
                       max_num_hands=MAX_HANDS,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5) as hands:
        
        # Collect all gesture classes
        classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
        print(f"Found {len(classes)} gesture classes")
        
        for folder in tqdm(classes, desc="Processing classes"):
            path = os.path.join(DATA_PATH, folder)
            label = folder.split('. ')[-1] if '. ' in folder else folder
            
            video_files = [f for f in os.listdir(path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if not video_files:
                print(f"No video files found in {path}")
                continue
                
            for vid in tqdm(video_files, desc=f"Processing {label} videos", leave=False):
                vid_path = os.path.join(path, vid)
                try:
                    # Process video with enhanced keypoint extraction
                    seq = process_video_enhanced(vid_path, hands)
                    
                    if seq.shape[0] == 0:
                        print(f"Warning: No frames extracted from {vid_path}")
                        continue
                        
                    if augment:
                        # Apply augmentation
                        aug_seqs = augment_keypoints(seq, augmentation_factor)
                        for aug_seq in aug_seqs:
                            seqs.append(aug_seq)
                            labels.append(label)
                    else:
                        seqs.append(seq)
                        labels.append(label)
                        
                    # Track class distribution
                    class_counts[label] = class_counts.get(label, 0) + 1
                except Exception as e:
                    print(f"Error processing {vid_path}: {e}")
    
    print("Class distribution:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} samples")
    
    # Pad sequences to max length
    max_len = max((s.shape[0] for s in seqs), default=0)
    print(f"Max sequence length: {max_len} frames")
    
    padded = []
    for s in seqs:
        pad = max_len - s.shape[0]
        if pad > 0:
            s = np.vstack([s, np.zeros((pad, s.shape[1]))])
        padded.append(s)
    
    X = np.array(padded, dtype='float32')
    y = np.array(labels)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} frames, {X.shape[2]} features")
    return X, y

def create_improved_model(input_shape, num_classes):
    """Build enhanced LSTM model with attention mechanism and bidirectional layers"""
    # Input and masking
    inputs = Input(shape=input_shape)
    masked = Masking(mask_value=0.)(inputs)
    
    # First Bidirectional LSTM block with residual connection
    x1 = Bidirectional(LSTM(64, return_sequences=True, activation='tanh', 
                            kernel_regularizer=l2(1e-5)))(masked)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.2)(x1)
    
    # Second Bidirectional LSTM block
    x2 = Bidirectional(LSTM(128, return_sequences=True, activation='tanh',
                           kernel_regularizer=l2(1e-5)))(x1)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    # Add attention mechanism to focus on important frames
    attn = MultiHeadAttention(num_heads=4, key_dim=64)(x2, x2)
    
    # Third Bidirectional LSTM block
    x3 = Bidirectional(LSTM(64, return_sequences=False, activation='tanh',
                           kernel_regularizer=l2(1e-5)))(attn)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.3)(x3)
    
    # Dense layers with increasing regularization
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-5))(x3)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_with_cross_validation():
    """Train the model with cross-validation for better generalization"""
    # Check for pre-processed data
    x_path = os.path.join(MODEL_DIR, 'X_padded.npy')
    y_path = os.path.join(MODEL_DIR, 'y_labels.npy')
    if os.path.exists(x_path) and os.path.exists(y_path):
        print("Loading pre-processed data...")
        X = np.load(x_path)
        y = np.load(y_path, allow_pickle=True)
    else:
        print("Processing dataset...")
        X, y = load_and_process_dataset(augment=AUGMENT_DATA)
        np.save(x_path, X)
        np.save(y_path, y, allow_pickle=True)
    
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_enc),
        y=y_enc
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=42)
    fold_accuracies = []
    best_model = None
    best_accuracy = 0
    
    # Save label encoder classes
    np.save(os.path.join(MODEL_DIR, 'label_encoder_classes.npy'), le.classes_)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Training Fold {fold+1}/{CROSS_VAL_FOLDS} ---")
        
        # Split data for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        
        # Create model
        model = create_improved_model((X_train.shape[1], X_train.shape[2]), len(le.classes_))
        
        # Setup callbacks
        temp_path = os.path.join(MODEL_DIR, f'temp_fold_{fold+1}_best.keras')
        callbacks = [
            ModelCheckpoint(temp_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=callbacks,
            class_weight=class_weight_dict
        )
        
        # Evaluate on test set
        loss, acc = model.evaluate(X_test, y_test)
        fold_accuracies.append(acc)
        print(f"Fold {fold+1} test accuracy: {acc:.4f}")
        
        # Save fold model
        fold_model_path = os.path.join(MODEL_DIR, f'fold_{fold+1}_model_{int(acc*100)}.keras')
        model.save(fold_model_path)
        
        # Track best model across folds
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
    
    # Print summary of cross-validation
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\nCross-validation results:")
    print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Save best model
    best_model_path = os.path.join(MODEL_DIR, f'best_model_{int(best_accuracy*100)}.keras')
    best_model.save(best_model_path)
    print(f"Best model saved to {best_model_path} with accuracy {best_accuracy:.4f}")
    
    return best_model, le, mean_acc

def predict_gesture(model, le, video_path):
    """Predict gesture with confidence score for a given video"""
    with mp_hands.Hands(static_image_mode=False,
                       max_num_hands=MAX_HANDS,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5) as hands:
        
        # Process video with enhanced features
        seq = process_video_enhanced(video_path, hands)
        
        if seq.shape[0] == 0:
            return "No hands detected", 0.0
            
        # Pad sequence to match model input shape
        max_len = model.input_shape[1]
        pad_len = max_len - seq.shape[0]
        if pad_len > 0:
            seq = np.vstack([seq, np.zeros((pad_len, seq.shape[1]))])
        elif pad_len < 0:
            # If sequence is longer than expected, truncate
            seq = seq[:max_len]
            
        # Make prediction
        Xp = seq.astype('float32')[None, ...]
        preds = model.predict(Xp, verbose=0)
        idx = np.argmax(preds[0])
        confidence = preds[0][idx]
        
        return le.classes_[idx], confidence

def visualize_training_history(history):
    """Plot training & validation accuracy and loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.show()

def evaluate_model_on_test_data(model, le, test_dir=None):
    """Evaluate model on separate test data if available"""
    if test_dir is None:
        test_dir = os.path.join(DATA_PATH, 'test')
        if not os.path.exists(test_dir):
            print("No separate test directory found")
            return
    
    correct = 0
    total = 0
    confusion = {}
    
    print(f"Evaluating model on test data in {test_dir}")
    
    for folder in os.listdir(test_dir):
        path = os.path.join(test_dir, folder)
        if not os.path.isdir(path):
            continue
            
        true_label = folder.split('. ')[-1] if '. ' in folder else folder
        if true_label not in confusion:
            confusion[true_label] = {'correct': 0, 'total': 0, 'predictions': {}}
            
        for vid in os.listdir(path):
            if not vid.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue
                
            vid_path = os.path.join(path, vid)
            pred_label, confidence = predict_gesture(model, le, vid_path)
            
            # Update confusion matrix
            if pred_label not in confusion[true_label]['predictions']:
                confusion[true_label]['predictions'][pred_label] = 0
            confusion[true_label]['predictions'][pred_label] += 1
            
            # Update counts
            confusion[true_label]['total'] += 1
            total += 1
            
            if pred_label == true_label:
                correct += 1
                confusion[true_label]['correct'] += 1
    
    # Print evaluation results
    if total > 0:
        accuracy = correct / total
        print(f"\nTest Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        # Print per-class accuracy
        print("\nPer-class accuracy:")
        for label, stats in confusion.items():
            if stats['total'] > 0:
                class_acc = stats['correct'] / stats['total']
                print(f"{label}: {class_acc:.4f} ({stats['correct']}/{stats['total']})")
                
                # Print top confusions for this class
                if stats['total'] - stats['correct'] > 0:
                    confusions = {k: v for k, v in stats['predictions'].items() if k != label}
                    if confusions:
                        top_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:3]
                        conf_str = ", ".join([f"{k} ({v})" for k, v in top_confusions])
                        print(f"  Top confusions: {conf_str}")
    else:
        print("No test data found")

if __name__ == '__main__':
    print("Starting sign language recognition model training...")
    start_time = time.time()
    
    # Train with cross-validation
    best_model, le, cv_accuracy = train_with_cross_validation()
    
    # Evaluate on separate test data if available
    evaluate_model_on_test_data(best_model, le)
    
    end_time = time.time()
    print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes")
    print(f"Cross-validation accuracy: {cv_accuracy:.4f}")
    print(f"Best model saved to {MODEL_DIR}")