import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
DATA_FILE        = "sequences.npy"                 # your extracted vectors
CLASS_FILE       = "label_encoder_classes.npy"     # your label encoder classes
MODEL_DIR        = "models"
BATCH_SIZE       = 16
EPOCHS           = 50
VALIDATION_SPLIT = 0.2
RANDOM_STATE     = 42

# ------------------------------------------------------------------------------
# 2. Utility functions
# ------------------------------------------------------------------------------
def reshape_for_3dcnn(X):
    """
    Reshape (N, seq_len, features) -> (N, seq_len, H, W, 1),
    choosing H×W = features and H as close to sqrt(features) as possible.
    """
    n, seq_len, feats = X.shape

    # find a good (H, W)
    for h in range(int(np.sqrt(feats)), 0, -1):
        if feats % h == 0:
            w = feats // h
            break

    X3 = X.reshape(n, seq_len, h, w, 1)
    print(f"Reshaped {X.shape} → {X3.shape} (seq_len={seq_len}, H×W={h}×{w})")
    return X3, (seq_len, h, w, 1)

def build_3dcnn(input_shape, num_classes):
    m = Sequential([
        Conv3D(32, (3,3,3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(), MaxPooling3D((1,2,2)), Dropout(0.2),

        Conv3D(64, (3,3,3), activation="relu", padding="same"),
        BatchNormalization(), MaxPooling3D((1,2,2)), Dropout(0.2),

        Conv3D(128, (3,3,3), activation="relu", padding="same"),
        BatchNormalization(), MaxPooling3D((2,2,2)), Dropout(0.3),

        Flatten(),
        Dense(256, activation="relu"), BatchNormalization(), Dropout(0.4),
        Dense(128, activation="relu"), BatchNormalization(), Dropout(0.4),
        Dense(num_classes, activation="softmax")
    ])
    m.compile(optimizer=Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

# ------------------------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------------------------
print("Loading data …")
X = np.load(DATA_FILE)            # shape: (n_samples, seq_len, n_features)
classes = np.load(CLASS_FILE)     # e.g. array of class names
y = np.arange(len(classes))       # if your classes file is ordered labels
# or, if you have a separate y.npy, do: y = np.load("labels.npy")

# ------------------------------------------------------------------------------
# 4. Prepare for training
# ------------------------------------------------------------------------------
X3, input_shape = reshape_for_3dcnn(X)
num_classes = len(classes)

X_train, X_test, y_train, y_test = train_test_split(
    X3, y, test_size=0.2, random_state=RANDOM_STATE
)

# ------------------------------------------------------------------------------
# 5. Build model & callbacks
# ------------------------------------------------------------------------------
model = build_3dcnn(input_shape, num_classes)
model.summary()

os.makedirs(MODEL_DIR, exist_ok=True)
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_3dcnn.keras"),
    monitor="val_accuracy", save_best_only=True, verbose=1
)
earlystop = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-5, verbose=1)

# ------------------------------------------------------------------------------
# 6. Train
# ------------------------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

# ------------------------------------------------------------------------------
# 7. Evaluate
# ------------------------------------------------------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {acc:.4f}, loss: {loss:.4f}")

# ------------------------------------------------------------------------------
# 8. Save final artifacts
# ------------------------------------------------------------------------------
model.save(os.path.join(MODEL_DIR, "final_3dcnn.keras"))
np.save(os.path.join(MODEL_DIR, "classes_saved.npy"), classes)
print("Saved model and classes.")
