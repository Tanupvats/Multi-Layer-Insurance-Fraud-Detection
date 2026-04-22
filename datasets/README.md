# `datasets/` — training data

Training data is **not committed**. Organize per-model as expected by
the scripts in `training/` (details in phase 3 commits).

Suggested layout:

```
datasets/
├── pose/
│   ├── train/{BL, BR, BS, FL, FR, FS, LS, RS}/*.jpg
│   └── val/{...}/*.jpg
├── siamese/                # identity triplets — structured per-car
│   └── {car_id_0001, car_id_0002, ...}/*.jpg
└── parts/                  # YOLOv11-seg format
    ├── images/{train,val}/*.jpg
    └── labels/{train,val}/*.txt
```
