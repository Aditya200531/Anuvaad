import numpy as np
import cv2
import mediapipe as mp
import os

# Initialize MediaPipe hand drawing tools
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
NUM_LANDMARKS = 21 * 3
DATA_PATH = "gesture_data"

# Create directories to store the data
def create_folder_structure(gestures):
    for gesture in gestures:
        os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

def collect_data(gestures):
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        for gesture in gestures:
            print(f"Collecting data for '{gesture}'... Press 's' to start, 'q' to stop when done.")
            count = 0
            recording = False

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks on the frame
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])

                        if len(landmarks) == NUM_LANDMARKS and recording:
                            np.save(os.path.join(DATA_PATH, gesture, f"{count}.npy"), np.array(landmarks))
                            count += 1
                            print(f"Saved frame {count} for '{gesture}'.")

                # Display the live video feed with landmarks
                cv2.putText(frame, f"Gesture: {gesture}, Samples: {count}", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('Data Collection', frame)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('s'):  # Press 's' to start recording
                    recording = True
                    print(f"Recording started for gesture '{gesture}'.")
                elif key == ord('q'):  # Press 'q' to stop recording the current gesture
                    recording = False
                    print(f"Stopped recording for '{gesture}' after {count} samples.")
                    break
                elif key == 27:  # Press 'ESC' to exit the data collection process
                    recording = False
                    print("Exiting data collection.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()

# Get number of gestures and gesture names from the user
def get_gestures():
    num_gestures = int(input("Enter the number of gestures you want to collect: "))
    gestures = []
    for i in range(num_gestures):
        gesture_name = input(f"Enter the name of gesture {i+1}: ")
        gestures.append(gesture_name)
    return gestures

# Main function to start the data collection process
if __name__ == "__main__":
    gestures = get_gestures()
    create_folder_structure(gestures)
    collect_data(gestures)
