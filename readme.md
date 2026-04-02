
# **AutoShield AI: Multi-Layer Insurance Fraud Detection**

**AutoShield AI** is a sophisticated Computer Vision pipeline designed to detect insurance fraud in vehicle claims. Unlike traditional systems that rely on metadata, this system uses **Geometric Invariance** and **Deep Identity Embeddings** to detect sophisticated manipulations like image mirroring and "Double Dipping" (reusing the same vehicle for multiple claims).

### **Multi-Layer Logic Gates**
1.  **Layer 1 (Pose Analysis):** Determines the vehicle's orientation (8-class model) to route the data to the correct segmentation model.
2.  **Layer 2 (component segmentation):** Segments the vehicle's components (wheels, headlight, hood, windshield etc) to route the data to the correct fraud detector.
2.  **Layer 3 (Geometric Consistency):** Uses **SuperGlue (GNN)** to detect mirrored/inverted images by decoupling the car from its background.
3.  **Layer 4 (Identity Verification):** Employs a **Siamese Network** focused on the windshield "fingerprint" to ensure vehicle uniqueness.

---
## System Architecture: Multi-Layer Fraud Guard
The system is built on a Modular Inference Pipeline architecture. By decoupling the various detection layers, we ensure that the system is both computationally efficient (using "early exit" logic) and highly maintainable.

[![System Design](Multi_layer_fraud_detection.png)]()

---

## **1. Project Structure**
```text
fraud_detection_poc/
├── main.py                 # System Orchestrator & Decision Engine
├── pipeline.py             # Inference Logic (YOLOv11 & SuperGlue)
├── models.py               # Neural Architectures (Siamese & Pose)
├── train_pose.py           # Training script for 8-class orientation
├── train_siamese.py        # Training script for Identity matching
├── train_segmentation.py   # Fine-tuning YOLOv11 for car parts
├── requirements.txt        # Production dependencies
└── car_parts.yaml          # YOLOv11 dataset configuration
```

---

## **2. Installation & Environment**
We recommend using Python 3.10+ and a CUDA-enabled GPU for production performance.

```bash
# Clone the repository
git clone https://github.com/Tanupvats/Multi-Layer-Insurance-Fraud-Detection.git
cd Multi-Layer-Insurance-Fraud-Detection

# Install dependencies
pip install -r requirements.txt
```

---

## **3. Model Training Procedures**

### **A. Pose Detection Model**
Trains an **EfficientNet-B0** to classify 8 orientations: `[FS, LS, RS, BS, FL, FR, BL, BR]`.
* **Key Constraint:** Data augmentation strictly excludes `RandomHorizontalFlip` to maintain label integrity.
* **Command:** Organize your data into 8 folders and run:
```bash
python train_pose.py 
```
[![Pose Model Training and Inference](pose_training_and_inference.png)]()

---

### **B. YOLOv11 Car Part Segmentation**
Fine-tunes **YOLOv11-seg** to isolate the windshield, headlights, and license plates.
* **Loss Function:** Optimized for Box, Mask, and Class loss.
* **Command:**
```bash
python train_segmentation.py 
```
[![Segmentation Model Training and Inference](yolov11_training-inference.png)]()

---

### **C. Siamese Identity Network**
Trains a **ResNet50** using **Triplet Margin Loss** to learn windshield embeddings.
* **Formula:** $L = \max(d(a, p) - d(a, n) + m, 0)$
* **Command:**
```bash
python train_siamese.py 
```
[![Siamese Model Training and Inference](Siamese_yolov11_training-inference.png)]()

---

## **4. Running the Inference Pipeline (`main.py`)**

The `main.py` script is the entry point. It executes the **Decoupling Logic**—splitting the car from the background to detect Photoshop manipulations.

### **How to Run:**
1.  Ensure you have your trained weights (`car_pose_v1.pth` and `siamese_identity.pth`) in the root.
2.  Place your claim images in a `data/` folder.
3.  Execute the orchestrator:

```bash
python main.py 
```

### **Inference Flow:**
1.  **Pose Check:** Identifies if Image A is a "Left Side" and Image B is a "Right Side."
2.  **Mirror Check:** If poses are complementary, the system flips Image A and runs **SuperGlue** on the **Background only**.
3.  **Identity Check:** YOLOv11 crops the windshield. Siamese calculates Cosine Similarity:
    $$\cos(\theta) = \frac{\mathbf{E_a} \cdot \mathbf{E_b}}{\|\mathbf{E_a}\| \|\mathbf{E_b}\|}$$
4.  **Verdict:** If similarity > 0.92, the system flags a "Duplicate Vehicle Identity."



---

## **5. Visualization & Audit Trail**
AutoShield AI is designed for **Explainable AI (XAI)**. Every fraud verdict generates a visualization file in the `outputs/` directory:
* **`viz_bg_mirror.jpg`**: Shows green correspondence lines between flipped backgrounds.
* **`viz_car_identity.jpg`**: Highlights the matching features in the windshield crops.

## Author

**Tanup Vats**  




