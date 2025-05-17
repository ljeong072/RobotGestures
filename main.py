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
from tkinter import *
from PIL import Image, ImageTk
from pickle import dump, load


class Gesture:
    def __init__(self, results):
        self.left = None
        self.right = None

        while results.multi_handedness:
            handedness = results.multi_handedness.pop()
            index = min(
                handedness.classification[0].index,
                len(results.multi_hand_landmarks) - 1,
            )
            if handedness.classification[0].label == "Right":
                right = np.array(
                    Gesture.parse_landmarks(results.multi_hand_landmarks[index])
                )
                self.right = right - right[0]
            else:
                left = np.array(
                    Gesture.parse_landmarks(results.multi_hand_landmarks[index])
                )
                self.left = left - left[0]

    @staticmethod
    def parse_landmarks(landmarks):
        return list(
            map(
                lambda landmark: (landmark.x, landmark.y, landmark.z),
                landmarks.landmark,
            )
        )

    def save(self, filename):
        with open(filename, "wb") as file:
            dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as file:
            return load(file)

    def get_difference(self, other):
        if (self.left is not None) and (other.left is not None):
            return np.mean(np.linalg.norm(self.left - other.left, axis=1))

        if (self.right is not None) and (other.right is not None):
            return np.mean(np.linalg.norm(self.right - other.right, axis=1))


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils

        self.width = 640
        self.height = 480
        self.is_camera_running = True
        self.frame_count = 0

        self.last_action_time = 0
        self.cooldown_seconds = 2

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.app = Tk()
        self.app.title("Gestures Capture")

        self.app.bind("<Escape>", lambda e: self.app.quit())

        self.label_widget = Label(self.app)
        self.label_widget.pack()

        self.button1 = Button(self.app, text="Open Camera", command=self.open_camera)
        self.button1.pack()
        self.button2 = Button(
            self.app, text="Capture Gesture", command=self.take_screenshot
        )
        self.button2.pack()
        self.button3 = Button(self.app, text="Open Gestures", command=self.switch_tab)
        self.button3.pack()

        self.open_camera()
        self.app.mainloop()

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def switch_tab(self):
        """Switch to the next tab"""
        self.is_camera_running = False

        new_window = Toplevel()
        new_window.title("New Window")
        # take saved gestures and print them all out

        gesture_files = os.listdir("./gestures")
        gesture_set = []
        for saved_gesture in gesture_files:
            gesture_set.append(Gesture.load(f"./gestures/{saved_gesture}"))

        print(gesture_set)

        # End of gesture functionality
        label = Label(new_window, text="This is a new window")
        label.pack()

        self.button1.config(text="Open Camera", command=self.open_camera)

    def save_gesture(self, frame):
        """Save the current frame as a PNG image"""
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gestures = os.listdir("./gestures")
        results = hands.process(self.frame)
        Gesture(results).save(f"./gestures/gesture{len(gestures)}.pkl")
        return

    def take_screenshot(self):
        """Take a screenshot of the current frame"""
        self.is_camera_running = False

        self.save_gesture(self.frame)

        self.button1.config(text="Open Camera", command=self.open_camera)

    def open_camera(self):
        """Start the camera capture process"""
        self.is_camera_running = True
        self.process_frame()
        self.button1.config(text="Close Camera", command=self.close_camera)

    def close_camera(self):
        """Stop the camera capture process"""
        self.is_camera_running = False
        self.button1.config(text="Open Camera", command=self.open_camera)

    def process_frame(self):
        """Process a single frame and schedule the next one"""
        if not self.is_camera_running:
            return

        ret, self.frame = self.cap.read()
        if ret:
            # Convert to RGB (not RGBA) for MediaPipe
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            display_frame = frame_rgb.copy()

            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                # Consider skipping some frames for drawing
                if self.frame_count % 2 == 0:  # Only draw every other frame
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                    current_time = time.time()

                    if is_open_palm(hand_landmarks):
                        if current_time - self.last_action_time > self.cooldown_seconds:
                            print("Gesture: Open Palm - Opening new tab")
                            # Send Ctrl+W or Cmd+W to browser via Selenium
                            browser_controller.quick_open_link(
                                "https://www.youtube.com"
                            )
                            self.last_action_time = current_time

                    elif is_fist(hand_landmarks):
                        if current_time - self.last_action_time > self.cooldown_seconds:
                            print("Gesture: Fist - Closing tab")
                            # Send Ctrl+T or Cmd+T to browser via Selenium
                            browser_controller.close_tab()

                            self.last_action_time = current_time
            # Use the frame with landmarks drawn on it
            captured_image = Image.fromarray(display_frame)

            # Convert captured image to photoimage
            photo_image = ImageTk.PhotoImage(image=captured_image)

            # Displaying photoimage in the label
            self.label_widget.photo_image = photo_image
            self.label_widget.configure(image=photo_image)

            self.frame_count += 1

            # Schedule the next frame
            self.app.after(50, self.process_frame)


class BrowserMacro:
    def __init__(self):
        # Setup Selenium WebDriver (make sure chromedriver is in PATH or specify path)
        self.driver = webdriver.Chrome()  # Or specify Service if needed
        self.driver.get("https://www.google.com")  # Open some page

        self.actions = ActionChains(self.driver)

    def __del__(self):
        self.driver.quit()

    def get_current_index(self):
        tabs = self.driver.window_handles
        current_tab = self.driver.current_window_handle
        return tabs.index(current_tab)

    def go_back_page(self):
        self.driver.back()

    def go_forward_page(self):
        self.driver.forward()

    def quick_open_link(self, link):
        self.driver.execute_script("window.open('');")
        self.driver.switch_to.window(self.driver.window_handles[-1])
        self.driver.get(link)

    def close_tab(self):
        self.driver.execute_script("window.close('');")
        self.driver.switch_to.window(self.driver.window_handles[-1])

    def get_difference(self, other):
        if (self.left is not None) and (other.left is not None):
            return np.mean(np.linalg.norm(self.left - other.left, axis=1))

        if (self.right is not None) and (other.right is not None):
            return np.mean(np.linalg.norm(self.right - other.right, axis=1))


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# modifier_key = Keys.COMMAND if platform.system() == "Darwin" else Keys.CONTROL

gesture_files = os.listdir("./gestures")
gesture_set = []
for saved_gesture in gesture_files:
    gesture_set.append(Gesture.load(f"./gestures/{saved_gesture}"))


def is_open_palm(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    open_fingers = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            open_fingers += 1
    return open_fingers >= 4


def is_fist(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    folded_fingers = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            folded_fingers += 1
    return folded_fingers >= 4


browser_controller = BrowserMacro()
app = App()
