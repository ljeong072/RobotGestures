datencity
puru_puru_pururin
Invisible

Hai — 5/6/25, 12:40 AM
oh
datencity — 5/6/25, 12:40 AM
the issue
Hai — 5/6/25, 12:40 AM
well then yoink anyones ig
datencity — 5/6/25, 12:40 AM
is that I think
I just don't wanna not get any credit
cause project is done
Hai — 5/6/25, 12:41 AM
bro
u shouldve locked in
datencity — 5/6/25, 12:41 AM
fuck I should've
what should I do
Hai — 5/6/25, 12:41 AM
uh
code clean upo and comments if it a code proj
datencity — 5/6/25, 12:42 AM
it is code project
Hai — 5/6/25, 12:42 AM
good
datencity — 5/6/25, 12:44 AM
should I lie and yoink contributions?
oh
should I yoink contributions
and kepe them vague
and wait until someone calls me out on it
Hai im scared
Hai — 5/6/25, 12:44 AM
bro
datencity — 5/6/25, 12:44 AM
I know nathan and another guy wouldn't report me
cause the other guy just made a commit for graphs
Hai — 5/6/25, 12:45 AM
Image
datencity — 5/6/25, 12:45 AM
am i cooked
be real
Hai — 5/6/25, 12:46 AM
ur chillin
who's that guy that put his dead pet in a box
it's not dead if u dont know it is or smth
i forgot
literally u
not a 0 grade until u see it
datencity — 5/6/25, 12:49 AM
shrodinger
datencity — 5/6/25, 12:49 AM
i h8 u
should i ask justin for advice
gpt said this
also the commits ._. if they check commits I'm fucked
could I not say that I was working on it, but people announced it as they went along
so me and the other guy
were kinda left behind
datencity — 5/6/25, 2:16 AM
hai
r u there
dont leave me
datencity — 12:17 PM
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)


    Gestures_set = []

    if len(Gestures_set) > 0:
        Gestures_set.append(results.multi_hand_landmarks)

        for i in len(Gestures_set):
            Current_comparison = Gestures_set[i] 

            for hand in Current_comparison:
                for jointIndex in range(21):
                    x = hand[jointIndex].x - results.hand[jointIndex].x# current gesture
                    y = hand[jointIndex].y - results.hand[jointIndex].y# current gesture
                    x = hand[jointIndex].z - results.hand[jointIndex].z# current gesture

                    if (x >= 0.02) and (y >= 0.02) and (z >= 0.02):
                        Gestures_set.append(results.multi_hand_landmarks)
                        break

    else:
        Gestures_set.append(results.multi_hand_landmarks)
datencity — 1:12 PM
 while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        current_hand = results.multi_hand_landmarks[0]

        if Gestures_set:
            for saved_hand_landmarks in Gestures_set:
                saved_hand = saved_hand_landmarks[0]  # First hand from saved gesture

                for joint_index in range(21):
                    x_diff = abs(saved_hand.landmark[joint_index].x - current_hand.landmark[joint_index].x)
                    y_diff = abs(saved_hand.landmark[joint_index].y - current_hand.landmark[joint_index].y)
                    z_diff = abs(saved_hand.landmark[joint_index].z - current_hand.landmark[joint_index].z)

                    if x_diff >= 0.02 or y_diff >= 0.02 or z_diff >= 0.02:
                        Gestures_set.append(results.multi_hand_landmarks)
                        print("New gesture added")
                        break
                break  # Compare only to the first saved gesture for now
        else:
            Gestures_set.append(results.multi_hand_landmarks)
            print("First gesture added")

        mp_drawing.draw_landmarks(frame, current_hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
Hai — 1:12 PM
class Gesture:
    def __init__(self, results):
        self.left = None
        self.right = None

        while results.multi_handedness:
            handedness = results.multi_handedness.pop()
            if handedness.classification[0].label == 'Right':
                self.right = Gesture.parse_landmarks(results.multi_hand_landmarks[handedness.classification[0].index])
            else:
                self.left = Guesture.parse_landmarks(results.multi_hand_landmarks[handedness.classification[0].index])

    @staticmethod
    def parse_landmarks(landmarks):
        return tuple(map(lambda landmark : (landmark.x, landmark.y, landmark.z), landmarks.landmark))

    def save(fliename):
    with open(filename, 'wb') as file:
        dump(self, file)

    @staticmethod
    def load(filename):
    with open(filename, 'rb') as file:
        return load(file)
Hai — 1:26 PM
class Gesture:
    def __init__(self, results):
        self.left = None
        self.right = None

        while results.multi_handedness:
            handedness = results.multi_handedness.pop()
            if handedness.classification[0].label == 'Right':
                self.right = Gesture.parse_landmarks(results.multi_hand_landmarks[handedness.classification[0].index])
            else:
                self.left = Gesture.parse_landmarks(results.multi_hand_landmarks[handedness.classification[0].index])

    @staticmethod
    def parse_landmarks(landmarks):
        return tuple(map(lambda landmark : (landmark.x, landmark.y, landmark.z), landmarks.landmark))

    def save(self, filename):
        with open(filename, 'wb') as file:
            dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return load(file)

    def get_difference(self, other):
        pass
datencity — 1:40 PM
import pickle
import mediapipe as mp
import time
import platform
import pyautogui
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
import cv2

from pickle import dump, load
class Gesture:
    def __init__(self, results):
        self.left = None
        self.right = None

        while results.multi_handedness:
            handedness = results.multi_handedness.pop()
            if handedness.classification[0].label == 'Right':
                self.right = Gesture.parse_landmarks(results.multi_hand_landmarks[handedness.classification[0].index])
            else:
                self.left = Gesture.parse_landmarks(results.multi_hand_landmarks[handedness.classification[0].index])

    @staticmethod
    def parse_landmarks(landmarks):
        return tuple(map(lambda landmark : (landmark.x, landmark.y, landmark.z), landmarks.landmark))

    def save(self, filename):
        with open(filename, 'wb') as file:
            dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return load(file)

    def get_difference(self, other):
        pass

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

modifier_key = Keys.COMMAND if platform.system() == 'Darwin' else Keys.CONTROL

# Setup Selenium WebDriver
driver = webdriver.Chrome()
driver.get("https://www.google.com")
actions = ActionChains(driver)

Gestures_set = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    print("HERE")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    gesture = Gesture(results)

    if gesture.right:  
        if Gestures_set:
            for saved_gesture in Gestures_set:
                match_found = True
                for joint_index in range(21):
                    x_diff = abs(saved_gesture.right[joint_index][0] - gesture.right[joint_index][0])
                    y_diff = abs(saved_gesture.right[joint_index][1] - gesture.right[joint_index][1])
                    z_diff = abs(saved_gesture.right[joint_index][2] - gesture.right[joint_index][2])

                    if x_diff >= 0.05 or y_diff >= 0.5 or z_diff >= 0.5:
                        match_found = False
                        break

                if not match_found:
                    Gestures_set.append(gesture)
                    print("New gesture added")
                    print(len(Gestures_set))
                    break
        else:
            Gestures_set.append(gesture)
            print("First gesture added")

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
driver.quit()
Hai — 1:46 PM
class Gesture:
    def __init__(self, results):
        self.left = None
        self.right = None

        while results.multi_handedness:
            handedness = results.multi_handedness.pop()
            index = min(handedness.classification[0].index, len(results.multi_hand_landmarks) - 1)
            if handedness.classification[0].label == 'Right':
                self.right = Gesture.parse_landmarks(results.multi_hand_landmarks[index])
            else:
                self.left = Gesture.parse_landmarks(results.multi_hand_landmarks[index])

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

    def get_difference(self, other):
        difference = [float('inf'), float('inf')]
        
        if self.left and other.left:
            selfleft = map(lambda position : position - self.left[0])
        if self.right and other.right:
            pass

        return difference
datencity — 2:05 PM
 import pickle
import mediapipe as mp
import time
import platform
import pyautogui
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
import cv2

from pickle import dump, load

class Gesture:
    def __init__(self, results):
        self.left = None
        self.right = None

        while results.multi_handedness:
            handedness = results.multi_handedness.pop()
            index = min(handedness.classification[0].index, len(results.multi_hand_landmarks) - 1)
            if handedness.classification[0].label == 'Right':
                self.right = Gesture.parse_landmarks(results.multi_hand_landmarks[index])
            else:
                self.left = Gesture.parse_landmarks(results.multi_hand_landmarks[index])

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

    def get_difference(self, other):
        difference = [float('inf'), float('inf')]
        
        if self.left and other.left:
            selfleft = map(lambda position : position - self.left[0])
        if self.right and other.right:
            pass

        return difference

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

modifier_key = Keys.COMMAND if platform.system() == 'Darwin' else Keys.CONTROL

# Setup Selenium WebDriver
driver = webdriver.Chrome()
driver.get("https://www.google.com")
actions = ActionChains(driver)

import os

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    gesture = Gesture(results)

    gesture_files = os.listdir("./gestures")
    gesture_set = []
    for saved_gesture in gesture_files:
    gesture_set.append(Gesture.load(saved_gesture))
        
    match_found = True
    for joint_index in range(21):
        x_diff = abs(saved_gesture.right[joint_index][0] - gesture.right[joint_index][0])
        y_diff = abs(saved_gesture.right[joint_index][1] - gesture.right[joint_index][1])
        z_diff = abs(saved_gesture.right[joint_index][2] - gesture.right[joint_index][2])

        if x_diff >= 0.05 or y_diff >= 0.5 or z_diff >= 0.5:
            match_found = False
            break

        if not match_found:
            gesture.save("./gesture/Gesture")
            #gesture.save("Gesture")
            #Gestures_set.append(gesture)
            print("New gesture added")
            print(len(Gestures_set))
            break
    else:
        gesture.save("./gesture/Gesture")
        #Gestures_set.append(gesture)
        print("First gesture added")



    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
driver.quit() 
Hai — 2:37 PM
import pickle
import mediapipe as mp
import os
import time
import platform
import pyautogui
Expand
main.py
4 KB
﻿
Hai
magicalhai
 
 
import pickle
import mediapipe as mp
import os
import time
import platform
import pyautogui
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
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
            if handedness.classification[0].label == 'Right':
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

    def get_difference(self, other):
        if (self.left is not None) and (other.left is not None):
            return np.mean(np.linalg.norm(self.left - other.left, axis=1))

        if (self.right is not None) and (other.right is not None):
            return np.mean(np.linalg.norm(self.right - other.right, axis=1))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

modifier_key = Keys.COMMAND if platform.system() == 'Darwin' else Keys.CONTROL

# Setup Selenium WebDriver
driver = webdriver.Chrome()
driver.get("https://www.google.com")
actions = ActionChains(driver)

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
        print(saved_gesture.get_difference(gesture))
                
    if len(gesture_set) == 0 and gesture.right is not None:
        gesture.save("./gestures/Gesture")
        gesture_set.append(gesture)
        print("First gesture added")



    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
driver.quit() 