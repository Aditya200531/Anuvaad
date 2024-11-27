import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Define parameters
NUM_FRAMES = 30  # Number of frames per gesture
NUM_LANDMARKS = 21 * 3  # 21 keypoints * (x, y, z) coordinates
DATA_PATH = "gesture_data"  # Path to gesture data

# Function to load gesture data from .npy files
def load_data():
    gestures = sorted(os.listdir(DATA_PATH))
    NUM_CLASSES = len(gestures)
    sequences, labels = [], []
    for gesture_idx, gesture in enumerate(gestures):
        gesture_folder = os.path.join(DATA_PATH, gesture)
        files = sorted(os.listdir(gesture_folder))
        for i in range(0, len(files) - NUM_FRAMES + 1, NUM_FRAMES):
            sequence = []
            for j in range(NUM_FRAMES):
                if i + j < len(files):
                    frame = np.load(os.path.join(gesture_folder, files[i + j]))
                    if frame.shape == (NUM_LANDMARKS,):
                        sequence.append(frame)
            if len(sequence) == NUM_FRAMES:
                sequences.append(np.array(sequence))
                labels.append(gesture_idx)
    return np.array(sequences), to_categorical(labels, num_classes=NUM_CLASSES), gestures

# Load and reshape data
X, y, gestures = load_data()
X = X.reshape(-1, NUM_FRAMES, NUM_LANDMARKS)

# Normalize the data
X = X / np.max(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(NUM_FRAMES, NUM_LANDMARKS)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    Conv1D(256, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    Flatten(),  # Flatten the output of the CNN for Dense layers
    
    Dense(128, activation='relu'),
    Dropout(0.3),
    
    Dense(len(gestures), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Save the model
model.save('isl_to_text_cnn_model.h5')

# Plot training accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
