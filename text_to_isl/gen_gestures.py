# file: hand_skeleton_visualizer.py

import os
import numpy as np
import pandas as pd
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Hand landmark connections for skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Folder containing gesture CSV files
GESTURE_FOLDER = "gesture_data"

def load_gesture_data(folder_path):
    """Load gesture data and compute average keypoints."""
    gesture_keypoints = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            gesture_name = os.path.splitext(file_name)[0]

            # Load CSV into a DataFrame
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path, header=None)

            # Calculate the average of each column
            average_keypoints = data.mean(axis=0).values.reshape(-1, 3)

            # Store in the dictionary
            gesture_keypoints[gesture_name] = average_keypoints

    return gesture_keypoints

def draw_hand_skeleton(image, keypoints):
    """Draw a 2D hand skeleton on an image."""
    h, w, _ = image.shape
    points = [(int(x * w), int(y * h)) for x, y, _ in keypoints]

    # Draw connections
    for connection in HAND_CONNECTIONS:
        start, end = connection
        cv2.line(image, points[start], points[end], (255, 0, 0), 2)

    # Draw keypoints
    for point in points:
        cv2.circle(image, point, 5, (0, 255, 0), -1)

    return image

def plot_3d_hand(keypoints):
    """Visualize a 3D hand skeleton using matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot keypoints
    xs, ys, zs = zip(*keypoints)
    ax.scatter(xs, ys, zs, c='r', marker='o')

    # Plot connections
    for connection in HAND_CONNECTIONS:
        start, end = connection
        ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], c='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def main():
    # Load gesture data
    print("Loading gesture data...")
    gesture_keypoints = load_gesture_data(GESTURE_FOLDER)
    print(f"Loaded gestures: {list(gesture_keypoints.keys())}")

    while True:
        # User input for gesture
        gesture_name = input("Enter a gesture name to visualize (or 'exit' to quit): ").strip().lower()

        if gesture_name == "exit":
            break

        if gesture_name not in gesture_keypoints:
            print(f"Gesture '{gesture_name}' not found. Available gestures: {list(gesture_keypoints.keys())}")
            continue

        # Retrieve the keypoints
        keypoints = gesture_keypoints[gesture_name]

        # Choose visualization mode
        mode = input("Choose visualization mode ('2d' or '3d'): ").strip().lower()

        if mode == "2d":
            # Create a blank canvas
            canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255

            # Draw the skeleton
            canvas = draw_hand_skeleton(canvas, keypoints)

            # Display the 2D visualization
            cv2.imshow(f"Gesture: {gesture_name}", canvas)
            print("Press any key to close the 2D visualization...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif mode == "3d":
            # Display the 3D visualization
            plot_3d_hand(keypoints)
        else:
            print("Invalid mode. Please choose '2d' or '3d'.")

if __name__ == "__main__":
    main()
