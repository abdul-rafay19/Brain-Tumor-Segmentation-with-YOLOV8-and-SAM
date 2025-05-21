# 🧠 Brain Tumor Detection & Segmentation using YOLOv8 + SAM

This project combines the powerful **YOLOv8** object detection model with Meta AI's **Segment Anything Model (SAM)** to detect and segment brain tumors from MRI scans. It processes medical images to identify tumor regions and generate precise segmentation masks for better analysis.

---

## 📌 Highlights

- 🔍 **Multi-object detection** in brain MRI scans using YOLOv8
- ✂️ **Fine-grained segmentation** of tumors using SAM
- 📸 **Output generation**: bounding boxes + polygonal masks
- 🧪 **Based on BraTS 2021 Dataset**
- ⚡ Can be extended with **Streamlit UI**

---

## 📁 Project Directory

```plaintext
PROJECT ARCH TECHNOLOGIES/
│
├── BraTS2021_Training_Data/         # Original MRI brain tumor data
│   └── (images, segmentations, etc.)
│
├── output/                          # Output after YOLO + SAM processing
│   ├── images/                      # Result images
│   ├── labels/                      # Bounding box labels
│   └── masks/                       # Mask polygons
│
├── yolo_brain_dataset/             # YOLO training dataset
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── YOLOV8_SAM/
│   ├── detect_multi_object_SAM.py   # Main script: YOLO detection + SAM masks
│   ├── appy.py                      # Streamlit app (optional)
│   ├── yolov8s.pt                   # Pretrained YOLOv8s model
│   ├── sam_vit_h_4b8939.pth         # SAM ViT-H checkpoint
│   ├── test_image.jpg               # Sample test image
│   ├── test_image2.jpg
│   ├── test_image3.jpg
│   ├── output_detection.jpg         # Output with bounding boxes + masks
│   ├── output_detectio2.jpg
│   ├── bounding_box_image1.txt      # YOLO bounding boxes
│   ├── bounding_box_image2.txt
│   ├── yolo_mask_image1.txt         # SAM masks in YOLO format
│   ├── yolo_mask_image2.txt
│   ├── visulise_mask.py             # Visualization utility
│   └── README.md
│
├── yolov8s.pt                       # YOLO model checkpoint
├── LICENSE
└── README.md

# 🧠 Sample MRI Image

**Original brain scan used as input:**

![Sample MRI](![image](https://github.com/user-attachments/assets/b3bde50f-115e-43ea-b43e-99ba0cac3451))

---

# 🧪 Sample Detection + Mask Output

**Result after YOLOv8 + SAM processing:**

![Output Detection]![image](https://github.com/user-attachments/assets/2cc13f20-c441-4764-8cfb-046f79280ecf))

---

## 🚀 How It Works

- **YOLOv8** detects multiple bounding boxes where tumors are likely present.
- **SAM** processes each bounding box and returns a detailed segmentation mask.
- **Masks** are saved as polygon coordinates for visualization and evaluation.

---

## 🧰 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/brain-tumor-yolo-sam.git
cd brain-tumor-yolo-sam

# Install dependencies
pip install torch torchvision torchaudio
pip install opencv-python
pip install ultralytics
pip install streamlit

## 🏃 Run the Project

### ➤ Run YOLO + SAM Pipeline:

```bash
cd YOLOV8_SAM
python detect_multi_object_SAM.py

### ➤ (Optional) Run Streamlit App

streamlit run appy.py

## 📦 Models Used

| Model      | Description                         | Source              |
|------------|-------------------------------------|---------------------|
| YOLOv8s    | Object detection for bounding boxes | Ultralytics YOLOv8 |
| SAM ViT-H  | Segmentation from bounding boxes    | Meta AI SAM         |

---

## 📚 Dataset

**BraTS 2021 (Brain Tumor Segmentation Challenge)**  
- MRI-based brain tumor scans  
- Converted to YOLO format in `yolo_brain_dataset/`  
- Segmentation masks extracted and saved in `output/masks/`

---

## 🔢 Sample Output Files

- `bounding_box_image1.txt` – YOLOv8 bounding boxes for test image  
- `yolo_mask_image1.txt` – Normalized polygon mask coordinates for the same image  
- `output_detection.jpg` – Annotated image (box + mask)

---

## 📷 Tested Sample from COCO

Also tested with general objects like this:

- Example YOLO detection class: **Class 74 = book**

---

## 👨‍💻 Developed By

**Abdul Rafay**  
📚 *BS Software Engineering | 🎯 AI & ML Enthusiast*  
🔗 [LinkedIn](https://www.linkedin.com/in/abdul-rafay19)  

---

## 📜 License

This repository is released under the **MIT License**.

---

## 🌟 Support & Contribution

If you found this project useful:

- ⭐ Star the repo  
- 🛠️ Fork and contribute  
- 📢 Share on LinkedIn and tag me!

