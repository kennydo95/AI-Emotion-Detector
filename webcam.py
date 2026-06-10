"""
webcam.py — Real-time emotion detection from webcam feed.
Run with: python src/webcam.py
Press Q to quit.
"""

import cv2
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from detector import analyse_frame, draw_results

WINDOW_NAME = "🎭 Emotion Detector  |  Press Q to quit"
PROCESS_EVERY_N_FRAMES = 5   # analyse every 5th frame to keep it smooth


def run_webcam(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Could not open webcam. Check your camera index.")
        return

    print("✅ Webcam opened. Press Q to quit.")
    frame_count = 0
    last_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to grab frame.")
            break

        frame_count += 1

        # Only run DeepFace every N frames (expensive operation)
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            last_faces = analyse_frame(frame)

        # Always draw latest results
        output = draw_results(frame, last_faces)

        # Show FPS hint
        cv2.putText(
            output,
            f"Faces: {len(last_faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2,
        )

        cv2.imshow(WINDOW_NAME, output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam closed.")


if __name__ == "__main__":
    run_webcam()
