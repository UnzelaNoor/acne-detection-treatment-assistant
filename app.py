"""
Streamlit App: Real-Time Acne Detection & Treatment Assistant
Uses YOLOv5 for acne detection and a local JSON knowledge base for chatbot responses.
"""

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
import json

# =======================
# Detection DataClass
# =======================
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    label: str

# =======================
# Load YOLO Model
# =======================
def load_yolo_model(weights_path="best.pt"):
    try:
        model = YOLO(weights_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# =======================
# Run YOLO Inference
# =======================
def run_yolo_on_image(model, img_path_or_array) -> List[Detection]:
    if model is None:
        return []

    try:
        results = model(img_path_or_array)  # can pass path or NumPy array
        dets: List[Detection] = []
        for result in results:  # results is a list for batch processing
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu().numpy()  # confidence
            labels = result.boxes.cls.cpu().numpy()   # class indices
            for i in range(len(boxes)):
                dets.append(
                    Detection(
                        bbox=(int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])),
                        score=float(scores[i]),
                        label="acne"  # assuming single class
                    )
                )
        return dets
    except Exception as e:
        st.error(f"Error during YOLO inference: {e}")
        return []

# =======================
# Draw Detections
# =======================
def draw_detections(pil_img: Image.Image, detections: List[Detection]) -> Image.Image:
    draw = ImageDraw.Draw(pil_img)
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        label = f"{det.label} {det.score:.2f}"
        draw.text((x1, max(0, y1 - 12)), label, fill=(255, 0, 0))
    return pil_img

# =======================
# Load Knowledge Base
# ======================= 

def load_knowledge_base(json_path="knowledge_base.json"):
    try:
        with open(json_path, "r") as f:
            kb = json.load(f)
        if isinstance(kb, dict):  # if JSON is a dict, convert to list of one dict
            kb = [kb]
        return kb
    except Exception as e:
        st.warning(f"Knowledge JSON not found or invalid at {json_path}. Chatbot will not work.\n{e}")
        return []
def get_treatment_advice(kb, query: str):
    if not kb:
        return "Knowledge base not loaded."
    query = query.lower()
    for entry in kb:
        if entry.get("problem", "").lower() in query:
            return entry.get("solution", "No solution found for this problem.")
    return "No advice found for this problem."

# =======================
# Streamlit App
# =======================
st.title("🧴 Acne Detection & Treatment Assistant")

# Sidebar input
with st.sidebar:
    st.header("Input")
    file = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)

# Load models and knowledge
yolo_model = load_yolo_model()
knowledge_base = load_knowledge_base()

# Run detection
if file:
    uploaded_img = Image.open(file).convert("RGB")
    np_img = np.array(uploaded_img)

    detections = run_yolo_on_image(yolo_model, np_img)

    # Apply confidence filter
    detections = [d for d in detections if d.score >= conf_thresh]

    st.write(f"✅ Final detections (after filtering): {len(detections)}")
    disp = uploaded_img.copy()
    disp = draw_detections(disp, detections)
    st.image(disp, caption="Annotated result", use_container_width=True)

    # Chatbot interaction
    user_query = st.text_input("Ask acne treatment advice:")
    if user_query:
        advice = get_treatment_advice(knowledge_base, user_query)
        st.success(advice)

else:
    st.info("Please upload an image to begin.")
