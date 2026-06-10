"""
app.py — Streamlit interface: upload a photo and detect emotions.
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from detector import analyse_frame, draw_results

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Emotion Detector",
    page_icon="🎭",
    layout="wide",
)

EMOTION_EMOJIS = {
    "happy": "😄", "sad": "😢", "angry": "😠",
    "surprise": "😲", "fear": "😨", "disgust": "🤢", "neutral": "😐",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎭 AI Emotion Detector")
st.markdown("Upload a photo with one or more faces and the AI will detect the **dominant emotion** for each person.")
st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📷 Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    # Convert to OpenCV BGR
    pil_img = Image.open(uploaded).convert("RGB")
    frame   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    col_orig, col_result = st.columns(2)

    with col_orig:
        st.subheader("Original")
        st.image(pil_img, use_column_width=True)

    with st.spinner("🔍 Detecting emotions..."):
        faces = analyse_frame(frame)
        output_bgr = draw_results(frame, faces)
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    with col_result:
        st.subheader("Detected Emotions")
        st.image(output_rgb, use_column_width=True)

    st.divider()

    # ── Per-face breakdown ────────────────────────────────────────────────────
    if not faces:
        st.warning("No faces detected. Try a clearer, well-lit photo.")
    else:
        st.subheader(f"Results — {len(faces)} face(s) found")
        for i, face in enumerate(faces):
            emotion = face["emotion"]
            emoji   = EMOTION_EMOJIS.get(emotion, "🙂")
            scores  = dict(sorted(face["scores"].items(), key=lambda x: -x[1]))

            with st.expander(f"Face {i+1}  {emoji}  **{emotion.upper()}**", expanded=True):
                # Top emotion
                st.metric("Dominant Emotion", f"{emoji} {emotion.capitalize()}", f"{scores[emotion]:.1f}%")

                # Bar chart for all emotions
                st.markdown("**All emotion scores:**")
                for em, score in scores.items():
                    bar_pct = int(score)
                    st.markdown(
                        f"`{em:<10}` "
                        f"{'█' * (bar_pct // 5)}"
                        f"{'░' * (20 - bar_pct // 5)}"
                        f"  {score:.1f}%"
                    )
else:
    # Placeholder guidance
    st.info("👆 Upload a photo above to get started. Works best with clear, front-facing photos.")
    st.markdown("""
    **Tips for best results:**
    - Use well-lit photos
    - Face should be clearly visible and not too small
    - Works with multiple faces in one photo
    - Supported formats: JPG, PNG
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Powered by DeepFace (pre-trained model) + OpenCV  |  No images are stored or sent to any server.")
