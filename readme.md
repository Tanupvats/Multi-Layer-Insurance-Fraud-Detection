
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
Multi-Layer-Insurance-Fraud-Detection/
├── configs/
│   └── model.yaml                         # thresholds, class maps, preprocessing
│
├── src/                                   # shared code — flat, not a package
│   ├── __init__.py                        # minimal marker
│   ├── schema.py                          # FraudReport, Verdict, Flag, Pose (Pydantic)
│   ├── config.py                          # Settings (env) + ModelConfig (YAML)
│   ├── image_io.py                        # EXIF-correct loader, bytes decoder, SHA256
│   ├── device.py                          # device resolution (no silent CPU fallback)
│   ├── logging_config.py                  # structured/JSON logging
│   ├── nets.py                            # CarPoseModel + SiameseNetwork (bugs fixed)
│   ├── pose_inferencer.py                 # one class per model
│   ├── car_segmenter.py                   #   COCO pretrained — car/bg split
│   ├── parts_segmenter.py                 #   CUSTOM trained — windshield (refuses w/o weights)
│   ├── feature_matcher.py                 #   SuperGlue keypoint matcher
│   ├── siamese_inferencer.py              #   ResNet50 windshield-identity embedder
│   └── pipeline.py                        # FraudPipeline — typed orchestrator → FraudReport
│
├── inference/                             # standalone per-model inference scripts
│   ├── _bootstrap.py                      # sys.path helper (no packaging)
│   ├── infer_pose.py                      # python inference/infer_pose.py --image x.jpg
│   ├── infer_parts.py                     # python inference/infer_parts.py --image x.jpg
│   ├── infer_matcher.py                   # python inference/infer_matcher.py --a a.jpg --b b.jpg
│   ├── infer_siamese.py                   # python inference/infer_siamese.py --a a.jpg --b b.jpg
│   └── infer_pipeline.py                  # full chain — replaces original main.py
│
├── api/                                   # FastAPI service
│   ├── __init__.py
│   ├── main.py                            # POST /analyze, GET /reports, /reports/{id}/viz/{kind}, /healthz, /readyz
│   ├── audit.py                           # SQLite audit log (WAL, thread-safe)
│   └── deps.py                            # pipeline + audit singletons (lru_cache)
│
├── training/                              # training scripts — one per model
│   ├── __init__.py
│   ├── README.md                          # run instructions + "why batch-hard mining"
│   ├── datasets.py                        # PoseDataset (no h-flip), TripletDataset
│   ├── utils.py                           # set_deterministic, EarlyStopper, batch_hard_triplet_loss,
│   │                                      #   pair_verification_accuracy, atomic checkpoint I/O
│   ├── train_pose.py                      # stratified split, weighted sampler, cosine LR, early stop, TB
│   ├── train_siamese.py                   # PK sampler + batch-hard triplet loss, identity-level val
│   └── train_segmentation.py              # YOLO with AdamW defaults + ONNX export
│
├── tests/                                 # pytest suite
│   ├── __init__.py
│   ├── conftest.py                        # generates synthetic fixture images (no committed PNGs)
│   ├── test_schema.py                     # Pydantic models + roundtrip JSON
│   ├── test_config.py                     # YAML loading + "pose classes sorted" validator
│   ├── test_image_io.py                   # EXIF rotation, corrupt/missing files, hashing
│   ├── test_pipeline_decisions.py         # pose-pair gate + verdict ladder (parametrized)
│   ├── test_audit.py                      # SQLite round-trip, latest-wins, verdict filter, persistence
│   └── test_training_utils.py             # batch-hard loss, EarlyStopper, cosine sim, verification
│
├── .github/
│   └── workflows/
│       └── ci.yml                         # ruff + pytest + docker build on push/PR
│
├── models/                                # trained weights (gitignored)
│   └── README.md                          # documents what goes here
│
├── datasets/                              # training data (gitignored)
│   └── README.md                          # suggested per-model layout
│
│└── outputs/                               # runtime artifacts (gitignored, created on demand)
│    ├── uploads/                           # content-addressed input images from API
│    └── <claim_id>/
│        ├── viz_bg_mirror.jpg              # background feature-match visualization
│        ├── windshield_a.jpg               # cropped windshield from image A
│        └── windshield_b.jpg               #   ...from image B
├── readme.md                              # project info
├── requirements.txt                       # flat deps (torch, fastapi, ultralytics, ...)
├── pytest.ini                             # pytest rootdir + warning filters
├── .env.example                           # documented operational config
├── .gitignore                             # excludes .env, models/*.pt, outputs/, audit.db, ...
├── .dockerignore                          # keeps build context small
├── Dockerfile                             # multi-stage, non-root, healthcheck
├── docker-compose.yml                     # local stack with mounted weights/outputs
├── car_parts.yaml                         # YOLO dataset config        
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
python training/train_pose.py 
```
[![Pose Model Training and Inference](pose_training_and_inference.png)]()

---

### **B. YOLOv11 Car Part Segmentation**
Fine-tunes **YOLOv11-seg** to isolate the windshield, headlights, and license plates.
* **Loss Function:** Optimized for Box, Mask, and Class loss.
* **Command:**
```bash
python training/train_segmentation.py 
```
[![Segmentation Model Training and Inference](yolov11_training-inference.png)]()

---

### **C. Siamese Identity Network**
Trains a **ResNet50** using **Triplet Margin Loss** to learn windshield embeddings.
* **Formula:** $L = \max(d(a, p) - d(a, n) + m, 0)$
* **Command:**
```bash
python training/train_siamese.py 
```
[![Siamese Model Training and Inference](Siamese_yolov11_training-inference.png)]()

---

## **4. Running the Inference Pipeline **

It executes the **Decoupling Logic**—splitting the car from the background to detect Photoshop manipulations.

### **How to Run:**
1.  Ensure you have your trained weights in models (`car_pose_v1.pth` and `siamese_identity.pth`) in the root.
2.  Place your claim images in a `dataset/` folder.
3.  Execute the orchestrator:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# --- Train ---
PYTHONPATH=. python training/train_pose.py          --data datasets/pose    --epochs 30
PYTHONPATH=. python training/train_siamese.py       --data datasets/siamese --epochs 40 --p 8 --k 4
PYTHONPATH=. python training/train_segmentation.py  --data car_parts.yaml   --epochs 100 --export-onnx

# --- Test ---
pytest -v                  # run the full suite locally
ruff check src api inference training tests

# --- Inference (unchanged from phase 2) ---
PYTHONPATH=. python inference/infer_pipeline.py --a a.jpg --b b.jpg --json

# --- Serve ---
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---


### **Inference Flow:**
1.  **Pose Check:** Identifies if Image A is a "Left Side" and Image B is a "Right Side."
2.  **Mirror Check:** If poses are complementary, the system flips Image A and runs **SuperGlue** on the **Background only**.
3.  **Identity Check:** YOLOv11 crops the windshield. Siamese calculates Cosine Similarity:
    $$\cos(\theta) = \frac{\mathbf{E_a} \cdot \mathbf{E_b}}{\|\mathbf{E_a}\| \|\mathbf{E_b}\|}$$
4.  **Verdict:** If similarity > 0.92, the system flags a "Duplicate Vehicle Identity."


## **5. Visualization & Audit Trail**
AutoShield AI is designed for **Explainable AI (XAI)**. Every fraud verdict generates a visualization file in the `outputs/` directory:
* **`viz_bg_mirror.jpg`**: Shows green correspondence lines between flipped backgrounds.
* **`viz_car_identity.jpg`**: Highlights the matching features in the windshield crops.

## Author

**Tanup Vats**  




