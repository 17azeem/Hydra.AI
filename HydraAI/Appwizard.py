import os
import cv2
import time
from cvzone.HandTrackingModule import HandDetector
import subprocess
import mediapipe as mp
import streamlit as st
import runpy


# Function to execute a command
def execute_command(command):
    os.system(command)


# Function to execute a command to launch an application
def launch_application(app_path, args=None):
    if args is None:
        args = []
    subprocess.Popen([app_path] + args, creationflags=subprocess.CREATE_NEW_CONSOLE)


# Function to execute a command to close an application
def close_application(process_name):
    subprocess.run(["taskkill", "/f", "/im", process_name])


# Paths to third-party application executables
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
chrome_process_name = "chrome.exe"

# Url's
url = "https://stackoverflow.com/"
url1 = "https://www.youtube.com/"

st.title("HYDRA.ai")
st.title("App Wizard")

st.image('img/appwizard.jpg',use_column_width='auto')



st.sidebar.title("Control Panel:")
# st.sidebar.markdown("Press here to launch")

start_button = st.sidebar.button("Launch your Assitant")

close_button = st.sidebar.button("Close Program")
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

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']

        current_time = time.time()

        # Gesture to launch Calculator (Thumb and Index fingers up)
        if fingers == [1, 1, 0, 0, 0]:
            if not calculator_launched and current_time - cooldown > 2:
                execute_command("calc")
                calculator_launched = True
                cooldown = current_time
                cv2.putText(img, "Launching Calculator...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

        # Gesture to close Calculator (Thumb and Pinky fingers up)
        elif fingers == [1, 0, 0, 0, 1]:
            if calculator_launched and current_time - cooldown > 2:
                # Attempt to close Calculator.exe
                close_application("Calculator.exe")
                # Also attempt to close ApplicationFrameHost.exe, which is often associated with UWP apps
                close_application("ApplicationFrameHost.exe")
                calculator_launched = False
                cooldown = current_time
                cv2.putText(img, "Closing Calculator...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

        # Gesture to launch Chrome (Stackoverflow)
        elif fingers == [1, 1, 1, 1, 1]:
            if not chrome_launched and current_time - cooldown > 2:
                launch_application(chrome_path, [url])
                chrome_launched = True
                cooldown = current_time
                cv2.putText(img, "Launching Chrome...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

        # Gesture to close Chrome (Thumb and Index fingers up)
        elif fingers == [1, 0, 1, 0, 0]:
            if chrome_launched and current_time - cooldown > 2:
                close_application(chrome_process_name)
                chrome_launched = False
                cooldown = current_time
                cv2.putText(img, "Closing Chrome...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
        # Gesture to launch Chrome (Stackoverflow)
        elif fingers == [0, 1, 1, 0, 0]:
            if not chrome_launched and current_time - cooldown > 2:
                launch_application(chrome_path, [url1])
                chrome_launched = True
                cooldown = current_time
                cv2.putText(img, "Launching Chrome...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                # Gesture to close Chrome (Thumb and Index fingers up)
        elif fingers == [1, 0, 1, 0, 0]:
            if chrome_launched and current_time - cooldown > 2:
                close_application(chrome_process_name)
                chrome_launched = False
                cooldown = current_time
                cv2.putText(img, "Closing Chrome...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

        # Gesture to launch Chrome (YouTube)
        elif fingers == [0, 1, 1, 0, 0]:
            if not chrome_launched and current_time - cooldown > 2:
                launch_application(chrome_path, [url1])
                chrome_launched = True
                cooldown = current_time
                cv2.putText(img, "Launching Chrome...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

        # close chrome(yt)
        elif fingers == [1, 0, 1, 0, 0]:
            if chrome_launched and current_time - cooldown > 2:
                close_application(chrome_process_name)
                chrome_launched = False

                cooldown = current_time
                cv2.putText(img, "Closing Chrome...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

    if close_button:
        break


cap.release()
cv2.destroyAllWindows()

if start_button:
    # Replace 'script_to_run.py' with the path to your Python script
    runpy.run_path(path_name='App.py')
