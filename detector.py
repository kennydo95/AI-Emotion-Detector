"""
detector.py — Face detection + emotion recognition using DeepFace.
Works on a single image (numpy array or file path).
"""

import cv2
import numpy as np
from deepface import DeepFace

# Emotion label → display colour (BGR for OpenCV)
EMOTION_COLORS = {
    "happy":    (0, 215, 255),    # gold
    "sad":      (255, 100, 50),   # blue-ish
    "angry":    (0, 0, 220),      # red
    "surprise": (0, 200, 200),    # yellow
    "fear":     (130, 0, 200),    # purple
    "disgust":  (0, 180, 80),     # green
    "neutral":  (180, 180, 180),  # grey
}
DEFAULT_COLOR = (255, 255, 255)


def analyse_frame(frame: np.ndarray) -> list[dict]:
    """
    Run DeepFace emotion analysis on a single BGR frame.

    Returns a list of dicts, one per detected face:
        {
            "emotion":    str,          # dominant emotion
            "scores":     dict,         # all emotion probabilities
            "box":        (x, y, w, h), # face bounding box
        }
    """
    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,   # don't crash if no face found
            silent=True,
        )
    except Exception:
        return []

    faces = []
    for r in results:
        region = r.get("region", {})
        faces.append({
            "emotion": r["dominant_emotion"],
            "scores":  r["emotion"],                 # dict of all emotions
            "box":     (
                region.get("x", 0),
                region.get("y", 0),
                region.get("w", 0),
                region.get("h", 0),
            ),
        })
    return faces


def draw_results(frame: np.ndarray, faces: list[dict]) -> np.ndarray:
    """Draw bounding boxes and emotion labels onto the frame."""
    output = frame.copy()
    for face in faces:
        emotion = face["emotion"]
        x, y, w, h = face["box"]
        color = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)

        # Bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        # Label background
        label = f"{emotion.upper()}  {face['scores'][emotion]:.0f}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(output, (x, y - lh - 12), (x + lw + 8, y), color, -1)

        # Label text
        cv2.putText(
            output, label,
            (x + 4, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 0), 2, cv2.LINE_AA,
        )
    return output
