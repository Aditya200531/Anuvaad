import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    SimpleRNN,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# MediaPipe Hands initialization
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils

# Parameters
DATA_PATH       = r"D:\ISL\GESTURES"
SEQUENCE_LENGTH = 30      # frames per gesture
MAX_HANDS       = 2
NUM_LANDMARKS   = 21
NUM_FEATURES    = 3       # x, y, z
EPOCHS          = 50
BATCH_SIZE      = 16

def extract_keypoints(frame, hands):
    """Extract hand keypoints from a frame using MediaPipe."""
    image  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    keypoints = np.zeros(MAX_HANDS * NUM_LANDMARKS * NUM_FEATURES)
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= MAX_HANDS:
                break
            for lm_idx, lm in enumerate(hand_landmarks.landmark):
                base = hand_idx * NUM_LANDMARKS * NUM_FEATURES + lm_idx * NUM_FEATURES
                keypoints[base    ] = lm.x
                keypoints[base + 1] = lm.y
                keypoints[base + 2] = lm.z
    return keypoints

def process_video(video_path, hands):
    """Process a video file and return a sequence of keypoints."""
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total = len(frames)
    if total <= SEQUENCE_LENGTH:
        idxs = list(range(total)) + [total-1] * (SEQUENCE_LENGTH - total)
    else:
        step = total / SEQUENCE_LENGTH
        idxs = [int(i * step) for i in range(SEQUENCE_LENGTH)]

    seq = []
    for i in idxs:
        seq.append(extract_keypoints(frames[i], hands))
    return np.array(seq)  # shape = (SEQUENCE_LENGTH, MAX_HANDS*NUM_LANDMARKS*NUM_FEATURES)

def load_dataset():
    """Load dataset and extract features."""
    X, y = [], []
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        for folder in sorted(os.listdir(DATA_PATH)):
            path = os.path.join(DATA_PATH, folder)
            if not os.path.isdir(path):
                continue
            name = folder.split('. ')[-1] if '. ' in folder else folder
            print(f"Processing gesture: {name}")
            for vid in os.listdir(path):
                if not vid.lower().endswith(('.mp4','.avi','.mov','.mkv')):
                    continue
                try:
                    seq = process_video(os.path.join(path, vid), hands)
                    X.append(seq)
                    y.append(name)
                except Exception as e:
                    print(f"  Error on {vid}: {e}")

    X = np.array(X)  # (n_samples, SEQUENCE_LENGTH, MAX_HANDS*NUM_LANDMARKS*NUM_FEATURES)
    y = np.array(y)
    return X, y

def create_rnn_model(input_shape, num_classes):
    model = Sequential([
        SimpleRNN(64,  return_sequences=True,  activation='tanh', input_shape=input_shape),
        Dropout(0.2),
        SimpleRNN(128, return_sequences=True,  activation='tanh'),
        Dropout(0.2),
        SimpleRNN(64,  return_sequences=False, activation='tanh'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    print("Loading dataset…")
    X, y = load_dataset()

    # Encode and split
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Input shape for RNN: {X_train.shape[1:]}, Classes: {len(le.classes_)}")

    model = create_rnn_model(X_train.shape[1:], len(le.classes_))

    # Callbacks
    ckpt = ModelCheckpoint(
        'best_rnn_model.keras',
        monitor='val_accuracy', verbose=1,
        save_best_only=True, mode='max'
    )
    es = EarlyStopping(
        monitor='val_accuracy', patience=10,
        restore_best_weights=True
    )

    print("Training RNN…")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[ckpt, es]
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"RNN Test accuracy: {acc:.4f}")

    # Plotting
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()

    plt.tight_layout()
    plt.savefig('rnn_training_history.png')

    # Save
    model.save('isl_gesture_rnn_model.keras')
    np.save('label_encoder_classes.npy', le.classes_)
    print("RNN model and label encoder saved.")
    return model, le, history

def predict_gesture(model, label_encoder, video_path):
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        seq = process_video(video_path, hands)
        X_input = seq.reshape(1, SEQUENCE_LENGTH, MAX_HANDS*NUM_LANDMARKS*NUM_FEATURES)
        pred = model.predict(X_input)[0]
        idx  = np.argmax(pred)
        return label_encoder.classes_[idx], pred[idx]

if __name__ == "__main__":
    model, le, history = train_model()
    # Example:
    # name, conf = predict_gesture(model, le, r"path\to\test_video.mp4")
    # print(f"Predicted {name} ({conf:.2%})")
