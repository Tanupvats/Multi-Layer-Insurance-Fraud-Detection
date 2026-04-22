import os
from ultralytics import YOLO

def train_car_parts_segmenter(data_yaml_path, epochs=100, imgsz=640):
    """
    Trains YOLOv11n-seg on custom car parts.
    Recommended Classes: [0: windshield, 1: headlight, 2: tire, 3: door, 4: hood]
    """
    
    model = YOLO("yolo11n-seg.pt") 

    
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,          
        device=0,          
        project="car_fraud_poc",
        name="part_segmenter_v1",
        optimizer="AdamW",
        lr0=0.01,
        augment=True,      
        val=True           
    )
    
    
    path = model.export(format="onnx")
    print(f"Model exported for production at: {path}")

if __name__ == "__main__":
    
    # train_car_parts_segmenter("car_parts.yaml")
    pass