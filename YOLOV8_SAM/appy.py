import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

# === Load Models Once ===
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8s.pt")
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return yolo_model, predictor

# === YOLO + SAM Inference Function ===
def detect_and_segment(image_pil, yolo_model, predictor):
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    image_rgb = np.array(image_pil)
    results = yolo_model(image_rgb, stream=True)

    boxes_list = []
    classes_list = []
    for result in results:
        boxes = result.boxes
        class_ids = result.boxes.cls.long().tolist()
        boxes_list.extend(boxes.xyxy.tolist())
        classes_list.extend(class_ids)

    if len(boxes_list) == 0:
        return image_bgr, "No objects detected by YOLO."

    input_boxes = torch.tensor([[int(i) for i in box] for box in boxes_list], device=yolo_model.device)
    predictor.set_image(image_rgb)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image_rgb.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    for i, mask in enumerate(masks):
        binary_mask = mask.squeeze().cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        color = (0, 255, 0)
        cv2.drawContours(image_bgr, contours, -1, color, 2)
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image_bgr, f"Class: {classes_list[i]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return image_bgr, "Detection + Segmentation complete."

# === Streamlit UI ===
st.title("YOLOv8 + SAM: Object Detection & Segmentation")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Original Image", use_column_width=True)

    with st.spinner("Running YOLOv8 + SAM..."):
        yolo_model, predictor = load_models()
        output_img, message = detect_and_segment(image_pil, yolo_model, predictor)
        st.success(message)
        st.image(output_img, caption="Output Image", use_column_width=True)
