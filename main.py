import cv2
import mediapipe as mp
import time
import platform
import pyautogui
import subprocess
import sys
from tkinter import *
from PIL import Image, ImageTk
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


modifier_key = Keys.COMMAND if platform.system() == "Darwin" else Keys.CONTROL


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils

        self.width = 1200
        self.height = 720
        self.is_camera_running = False

        self.last_action_time = 0
        self.cooldown_seconds = 2

        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.app = Tk()
        self.app.title("Gestures Capture")

        self.app.bind("<Escape>", lambda e: self.app.quit())

        self.label_widget = Label(self.app)
        self.label_widget.pack()

        self.button1 = Button(self.app, text="Open Camera", command=self.open_camera)
        self.button1.pack()

        self.app.mainloop()

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

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

        ret, frame = self.cap.read()
        if ret:
            # Convert to RGB (not RGBA) for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
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

                            browser_controller.actions.key_down(modifier_key).send_keys(
                                "w"
                            ).key_up(modifier_key).perform()
                            self.last_action_time = current_time

        # Capture the latest frame and transform to image
        captured_image = Image.fromarray(frame_rgb)

        # Convert captured image to photoimage
        photo_image = ImageTk.PhotoImage(image=captured_image)

        # Displaying photoimage in the label
        self.label_widget.photo_image = photo_image

        # Configure image in the label
        self.label_widget.configure(image=photo_image)

        # Schedule the next frame processing after 10ms
        self.app.after(10, self.process_frame)


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

    def switch_previous_tab(self):
        previous_index = (self.get_current_index() - 1) % len(
            self.driver.window_handles
        )
        self.driver.switch_to.window(self.driver.window_handles[previous_index])

    def switch_next_tab(self):
        next_index = (self.get_current_index() + 1) % len(self.driver.window_handles)
        self.driver.switch_to.window(self.driver.window_handles[next_index])

    def open_app(self, app_name):
        try:
            if sys.platform == "win32":
                subprocess.run([app_name], check=True)
            elif sys.platform == "darwin":
                subprocess.run(["open", app_name], check=True)
            elif sys.platform.startswith("linux"):
                subprocess.run([app_name], check=True)
            else:
                print(f"Unsupported operating system: {sys.platform}")
                return
        except FileNotFoundError:
            print(f"Error: Application '{app_name}' not found.")
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to open '{app_name}'. Return code: {e.returncode}")


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

"""
import cv2
import mediapipe as mp
import pyautogui
import time
import platform

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_action_time = 0
cooldown_seconds = 2
modifier_key = 'command' if platform.system() == 'Darwin' else 'ctrl'

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            current_time = time.time()

            if is_open_palm(hand_landmarks):
                if current_time - last_action_time > cooldown_seconds:
                    print("Gesture: Open Palm - Opening new tab")
                    pyautogui.hotkey(modifier_key, 't')  # Open new tab
                    last_action_time = current_time

            elif is_fist(hand_landmarks):
                if current_time - last_action_time > cooldown_seconds:
                    print("Gesture: Fist - Closing tab")
                    pyautogui.hotkey(modifier_key, 'w')  # Close tab
                    last_action_time = current_time

    cv2.imshow("Hand Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if is_open_palm(hand_landmarks):
    pyautogui.hotkey('ctrl', 't')  # open new tab in active browser

if is_fist(hand_landmarks):
    pyautogui.hotkey('ctrl', 'w')  # close current tab

cap.release()
cv2.destroyAllWindows()
"""


"""
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with hand landmarks
    cv2.imshow("Hand Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

known_gestures = {}  # Dictionary: gesture_name -> list of known landmarks

def load_known_gestures(data_path="gestures_data"):
    known_gestures = {}
    if not os.path.exists(data_path):
        return known_gestures

    for gesture_name in os.listdir(data_path):
        gesture_folder = os.path.join(data_path, gesture_name)
        landmarks = []
        for file in os.listdir(gesture_folder):
            if file.endswith(".npy"):
                landmarks.append(np.load(os.path.join(gesture_folder, file)))
        known_gestures[gesture_name] = landmarks
    return known_gestures



# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
"""
