import torch
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# === YOLOv8 Detection Function ===
def yolov8_detection(model, image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb, stream=True)

    boxes_list = []
    classes_list = []
    for result in results:
        boxes = result.boxes
        class_ids = result.boxes.cls.long().tolist()
        boxes_list.extend(boxes.xyxy.tolist())
        classes_list.extend(class_ids)

    bbox = [[int(i) for i in box] for box in boxes_list]
    return bbox, classes_list, image_rgb, image

# === Load YOLOv8 Model ===
model = YOLO(r"D:\Downloads\Project Arch Technlogies\YOLOV8_SAM\yolov8s.pt")

# === Use Your Test Image Path ===
image_path = r"D:\Downloads\Project Arch Technlogies\YOLOV8_SAM\test_image2.jpg"

# === Run Detection ===
yolov8_boxes, yolov8_class_ids, image_rgb, image_bgr = yolov8_detection(model, image_path)

# === Skip SAM if YOLO finds nothing ===
if len(yolov8_boxes) == 0:
    print("No objects detected by YOLO. Skipping SAM.")
    exit()

input_boxes = torch.tensor(yolov8_boxes, device=model.device)

# === Load SAM ===
sam_checkpoint = r"D:\Downloads\Project Arch Technlogies\YOLOV8_SAM\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# === Predict with SAM ===
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)
transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image_rgb.shape[:2])

masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

# === Save Outputs ===
for i, mask in enumerate(masks):
    binary_mask = mask.squeeze().cpu().numpy().astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # === Save YOLO Format Bounding Box ===
    with open(r"D:\Downloads\Project Arch Technlogies\YOLOV8_SAM\bounding_box_image2.txt", "a") as f:
        f.write(
            "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                yolov8_class_ids[i],
                (x + w / 2) / image_rgb.shape[1],
                (y + h / 2) / image_rgb.shape[0],
                w / image_rgb.shape[1],
                h / image_rgb.shape[0],
            )
        )

    # === Save Segmentation Mask ===
    mask_pts = largest_contour.reshape(-1, 2)
    mask_norm = mask_pts / np.array([image_rgb.shape[1], image_rgb.shape[0]])

    with open(r"D:\Downloads\Project Arch Technlogies\YOLOV8_SAM\yolo_mask_image2.txt", "a") as f:
        f.write(f"{yolov8_class_ids[i]}")
        for pt in mask_norm:
            f.write(" {:.6f} {:.6f}".format(pt[0], pt[1]))
        f.write("\n")

    print("Bounding Box:", [x, y, w, h])
    print("YOLO Mask Points:", mask_norm.reshape(-1))

# === Save Annotated Output Image ===
cv2.imwrite(r"D:\Downloads\Project Arch Technlogies\YOLOV8_SAM\output_detectio2.jpg", image_bgr)
print("Output image saved successfully!")
