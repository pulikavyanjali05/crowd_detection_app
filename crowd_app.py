import streamlit as st
import cv2
from ultralytics import YOLO
import torch
import tempfile
import time

# Title
st.title("👥 Real-Time Crowd Detection App with Alarm & Webcam")

# Load YOLO model
model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Sidebar controls
st.sidebar.header("⚙️ Settings")
crowd_limit = st.sidebar.slider("Set Crowd Limit", 1, 100, 10)
use_webcam = st.sidebar.toggle("Use Webcam")
alarm_enabled = st.sidebar.toggle("Enable Alarm")

# Upload video if not using webcam
video_file = None
if not use_webcam:
    video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Start button
start_button = st.button("▶️ Start Detection")

# Alarm state (to ensure it plays only once)
alarm_triggered = False

def play_alarm_once():
    """Play alarm sound once."""
    st.warning("🚨 Overcrowded! Alarm Triggered!")
    st.audio("https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg")

if start_button:
    stframe = st.empty()
    count_placeholder = st.empty()

    # Open video source
    if use_webcam:
        cap = cv2.VideoCapture(0)
        st.info("📷 Using webcam for live detection...")
    else:
        if video_file is None:
            st.error("Please upload a video file or enable webcam.")
            st.stop()
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())
        cap = cv2.VideoCapture(temp_file.name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)
        boxes = results[0].boxes.xyxy
        crowd_count = len(boxes)

        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Display crowd count
        count_placeholder.markdown(f"### 👥 Total People Detected: `{crowd_count}`")

        # Alarm logic — play once when limit is crossed
        if alarm_enabled:
            if crowd_count > crowd_limit and not alarm_triggered:
                play_alarm_once()
                alarm_triggered = True
            elif crowd_count <= crowd_limit:
                alarm_triggered = False  # Reset when crowd reduces

        # Show frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        time.sleep(0.05)

    cap.release()
    st.success("✅ Detection finished!")
