import cv2
import time
import streamlit as st
import math
import pyautogui
import mediapipe as mp
import screen_brightness_control as sbc
from cvzone.HandTrackingModule import HandDetector
import subprocess
import runpy


pyautogui.FAILSAFE = False


# Function to execute a command
def launch_application(app_path, args=None):
    if args is None:
        args = []
    subprocess.Popen([app_path] + args)


# mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Flags to track the state of applications
calculator_launched = False
chrome_launched = False
cooldown = time.time()

# Hand gesture detection
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initial distance and brightness
initial_distance = None
brightness = sbc.get_brightness(display=0)[0]  # Get brightness of the first display

st.title("Hand Gesture Control")

st.sidebar.title("Control Panel:")

start_button = st.sidebar.button("Launch your Assistant")
# Add a close button using Streamlit
close_button = st.sidebar.button("Close Program")

# Video stream placeholder
video_placeholder = st.empty()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    img = cv2.flip(img, 1)
    hands_list, img = detector.findHands(img)

    if hands_list:
        hand = hands_list[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']

        current_time = time.time()

        # Volume controller and brightness control
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of thumb tip and index fingertip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                thumb_tip_x, thumb_tip_y = thumb_tip.x, thumb_tip.y
                index_finger_tip_x, index_finger_tip_y = index_finger_tip.x, index_finger_tip.y

                # Calculate the distance between thumb tip and index fingertip
                distance = math.hypot(index_finger_tip_x - thumb_tip_x, index_finger_tip_y - thumb_tip_y)

                if initial_distance is None:
                    initial_distance = distance

                # Adjust brightness based on the distance change
                brightness_change = (distance - initial_distance) * 1000  # Adjust the multiplier as needed
                new_brightness = int(brightness + brightness_change)
                new_brightness = max(0, min(100, new_brightness))  # Clamp brightness to [0, 100]

                sbc.set_brightness(new_brightness, display=0)
                brightness = new_brightness  # Update the current brightness

                # Volume control based on finger positions
                if index_finger_tip_y < thumb_tip_y:
                    pyautogui.press('volumeup')
                elif index_finger_tip_y > thumb_tip_y:
                    pyautogui.press('volumedown')

    # Check if close button is pressed

    if close_button:
        break
    # video_placeholder.image(img)

    # Update the video frame every 10 milliseconds (adjust as needed)
    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()

if start_button:
    # Replace 'script_to_run.py' with the path to your Python script
    runpy.run_path(path_name='vol-controler.py')
