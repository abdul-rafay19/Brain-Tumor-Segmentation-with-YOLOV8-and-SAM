# 🧠 Brain Tumor Detection with YOLOv8 + SAM 🔬

An advanced object detection and segmentation pipeline that leverages **YOLOv8** for tumor localization and **SAM (Segment Anything Model)** for precise mask generation. Developed using PyTorch, OpenCV, and integrated with an optional Streamlit UI for real-time interaction.

---

## 📌 Overview

This project combines the power of **YOLOv8** (by Ultralytics) and **SAM** (by Meta AI) to detect and segment brain tumors from MRI scans.

- 🔍 Detects multiple objects with YOLOv8  
- 🧠 Generates pixel-accurate tumor masks using SAM  
- 💾 Outputs bounding box and segmentation coordinates  
- 📊 Supports general object detection from COCO too  

---

## 🧪 Sample Outputs

### 🎯 Input MRI Scan

> Original brain scan used as input:

(https://github.com/user-attachments/assets/577ee8e7-4245-4bbe-bead-410549c36faf)

### ✅ Output: Detected Tumor(s)

> After YOLOv8 detection + SAM segmentation:

(https://github.com/user-attachments/assets/948c1820-8a76-431a-bf71-cf0e66a2b9f8)

---

## 🚀 How It Works

1. **YOLOv8** detects all bounding boxes of interest (e.g., tumors).  
2. Each bounding box is passed to **SAM**, which generates a detailed segmentation mask.  
3. Both bounding boxes and polygonal masks are saved in `.txt` files for further use or annotation.  

---

## 🧰 Installation

Clone the repository and install the necessary libraries:

```bash
git clone https://github.com/your-username/brain-tumor-yolo-sam.git
cd brain-tumor-yolo-sam
```

### 🔧 Install Python Dependencies:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install ultralytics
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### ▶️ (Optional) Install Streamlit:

```bash
pip install streamlit
```

---

## 🏃 Run the Project

### ➤ Run YOLO + SAM Pipeline:

```bash
cd YOLOV8_SAM
python detect_multi_object_SAM.py
```

### ➤ (Optional) Run Streamlit App:

```bash
streamlit run appy.py
```

---

## 📦 Models Used

| Model       | Description                         | Source                   |
|-------------|-------------------------------------|---------------------------|
| YOLOv8s     | Object detection for bounding boxes | Ultralytics YOLOv8        |
| SAM ViT-H   | Segmentation from bounding boxes    | Meta AI Segment Anything  |

---

## 📚 Dataset

- **BraTS 2021**: Brain Tumor Segmentation Challenge  
- MRI-based brain tumor scans  
- Converted to YOLOv8 format inside `yolo_brain_dataset/`  
- Masks saved in `output/masks/`  

---

## 🔢 Output Files

| File Name               | Description                                      |
|-------------------------|--------------------------------------------------|
| `bounding_box_image1.txt` | YOLOv8 bounding boxes for test image          |
| `yolo_mask_image1.txt`    | Normalized mask polygon coordinates           |
| `output_detection.jpg`    | Annotated image with box + mask overlay       |

---

## 📷 Tested on COCO Too!

The pipeline also works on general object detection tasks. For example:

- Detected: **Class 74 = Book**

You can try feeding other images to test its generalizability.

---

## 👨‍💻 Developed By

**Abdul Rafay**  
📚 BS Software Engineering | 🎯 AI & ML Enthusiast   
🔗 [LinkedIn](https://www.linkedin.com/in/abdul-rafay19)

---

## 📜 License

This repository is licensed under the **MIT License**.

---

## 🌟 Support & Contribution

If you found this helpful:

- ⭐ Star the repo  
- 🍴 Fork it and contribute  
- 📢 Share on LinkedIn and tag me!  

---

> 🔍 Accurate detection. 🎯 Precise segmentation. 🚀 Built with passion.
