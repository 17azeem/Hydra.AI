import os
import cv2
import streamlit as st
from pdf2image import convert_from_path
from cvzone.HandTrackingModule import HandDetector


# Function to convert PDF to images
def convert_pdf_to_images(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    images = []
    for i, page in enumerate(pages):
        image_path = f'page_{i}.jpg'
        page.save(image_path, 'JPEG')
        images.append(image_path)
    return images


# Streamlit interface
st.set_page_config(page_title="Gesture Controlled Presentation", layout="wide")

st.markdown("""
    <style>
     .main {
        color:#ffffff;
        background-color: #000000;
        padding: 10px;
        border-radius: 10px;
    }

    .block-container {
        background-color:#000000;
        padding: 0 !important;
    }
    .css-18e3th9 {
        padding: 0 !important;
    }
    .css-1d391kg {
        padding: 0 !important;
    }
    .css-1aumxhk {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    </style>
    """, unsafe_allow_html=True)


st.sidebar.title("Navigation")
st.sidebar.markdown("Use the sidebar to navigate through the app.")
st.image('img/npresent.png',use_column_width='auto')
pdf_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
click = False
counter = 0
delay=35
if pdf_file:
    pdf_path = pdf_file.name
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    st.sidebar.write("Converting PDF to images...")
    presentation_images = convert_pdf_to_images(pdf_path)
    total_pages = len(presentation_images)

    st.sidebar.write(f"Total pages: {total_pages}")

    slide_number = st.sidebar.number_input("Slide Number", min_value=1, max_value=total_pages, value=1)
    image_path = presentation_images[slide_number - 1]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption=f"Slide {slide_number}", use_column_width=True)

    st.sidebar.write("You can control the slides using hand gestures.")

    # Hand gesture detection
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            cx, cy = hand['center']

            if fingers == [1, 0, 0, 0, 0]:  # Left gesture
                if slide_number > 1:
                    slide_number -= 1
            elif fingers == [0, 1, 0, 0, 0]:  # Right gesture
                if slide_number < total_pages:
                    slide_number += 1

        image_path = presentation_images[slide_number - 1]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        stframe.image(image, caption=f"Slide {slide_number}", use_column_width=True)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    if click:
        counter += 1
        if counter > delay:
            counter = 0
            click=False
  
    cap.release()
    cv2.destroyAllWindows()
