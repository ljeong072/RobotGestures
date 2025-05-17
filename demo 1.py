import mediapipe as mp
import os
import platform
import cv2
import numpy as np

from pickle import dump, load

class Gesture:
    def __init__(self, results):
        self.left = None
        self.right = None

        while results.multi_handedness:
            handedness = results.multi_handedness.pop()
            index = min(handedness.classification[0].index, len(results.multi_hand_landmarks) - 1)
            if handedness.classification[0].label != 'Right':
                right = np.array(Gesture.parse_landmarks(results.multi_hand_landmarks[index]))
                self.right = right - right[0]
            else:
                left = np.array(Gesture.parse_landmarks(results.multi_hand_landmarks[index]))
                self.left = left - left[0]

    @staticmethod
    def parse_landmarks(landmarks):
        return list(map(lambda landmark : (landmark.x, landmark.y, landmark.z), landmarks.landmark))

    def save(self, filename):
        with open(filename, 'wb') as file:
            dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return load(file)
        
    def is_comparable(self, other):
        return (self.left is None) == (other.left is None) and (self.right is None) == (other.right is None)

    def get_difference(self, other):
        left_diff = right_diff = 0
        
        if (self.left is not None) and (other.left is not None):
            left_diff = np.mean(np.linalg.norm(self.left - other.left, axis=1))

        if (self.right is not None) and (other.right is not None):
            right_diff = np.mean(np.linalg.norm(self.right - other.right, axis=1))

        return left_diff + right_diff

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_files = os.listdir("./gestures")
gesture_set = []
for saved_gesture in gesture_files:
    gesture_set.append(Gesture.load(f"./gestures/{saved_gesture}"))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    gesture = Gesture(results)
        
    match_found = True
    for saved_gesture in gesture_set:
        if gesture.is_comparable(saved_gesture):
            print(saved_gesture.get_difference(gesture))
                
    # if len(gesture_set) == 0 and gesture.right is not None:
    #     gesture.save("./gestures/Gesture")
    #     gesture_set.append(gesture)
    #     print("First gesture added")
    if cv2.waitKey(1) & 0xFF == ord("s"):
        gesture_set = [gesture]
        print("gesture set")

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
