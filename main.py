import cv2
import torch
import numpy as np
from pathlib import Path


from pipeline import FraudDetectionEngine
from models import SiameseNetwork, CarPoseModel 

class ProductionFraudOrchestrator:
    def __init__(self, pose_weights="car_pose_v1.pth", siamese_weights="siamese_identity.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.engine = FraudDetectionEngine()
        
        
        self.pose_meta = torch.load(pose_weights, map_map=self.device)
        self.pose_model = CarPoseModel(num_classes=len(self.pose_meta['classes']))
        self.pose_model.load_state_dict(self.pose_meta['model_state_dict'])
        self.pose_model.to(self.device).eval()
        
        
        self.siamese = SiameseNetwork().to(self.device).eval()
        if Path(siamese_weights).exists():
            self.siamese.load_state_dict(torch.load(siamese_weights, map_location=self.device))

    def predict_pose(self, image):
        """Standardized pose inference."""
        img_t = cv2.resize(image, (224, 224))
        img_t = torch.from_numpy(img_t).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.pose_model(img_t)
            idx = torch.argmax(logits, dim=1).item()
        return self.pose_meta['classes'][idx]

    def extract_windshield_crop(self, image):
        """Uses the YOLOv11 engine to find and crop the windshield."""
        results = self.engine.segmenter(image, verbose=False)
        for result in results:
            if result.boxes is not None:
                for i, cls in enumerate(result.boxes.cls):
                    if int(cls) == 0: # Windshield class from our YAML
                        box = result.boxes.xyxy[i].cpu().numpy().astype(int)
                        # Tight crop for identity matching
                        return image[box[1]:box[3], box[0]:box[2]]
        return None

    def analyze_claim(self, img_path_a, img_path_b):
        """
        Main Decision Logic for Fraud Detection.
        """
        report = {"verdict": "CLEAN", "flags": [], "scores": {}}
        
        img_a = cv2.imread(img_path_a)
        img_b = cv2.imread(img_path_b)
        
        # 1. Detect Poses
        pose_a = self.predict_pose(img_a)
        pose_b = self.predict_pose(img_b)
        report["poses"] = {"img_a": pose_a, "img_b": pose_b}

        
        if (pose_a == "LS" and pose_b == "RS") or (pose_a == "RS" and pose_b == "LS"):
            img_a_flipped = cv2.flip(img_a, 1)
            car_a, bg_a = self.engine.segment_car_and_bg(img_a_flipped)
            car_b, bg_b = self.engine.segment_car_and_bg(img_b)
            
            
            bg_score = self.engine.match_features(bg_a, bg_b, "viz_bg_mirror.jpg")
            report["scores"]["mirror_bg_matches"] = bg_score
            
            if bg_score > 80: 
                report["verdict"] = "FRAUD"
                report["flags"].append("INVERTED_IMAGE_DETECTION")

        
        ws_a = self.extract_windshield_crop(img_a)
        ws_b = self.extract_windshield_crop(img_b)
        
        if ws_a is not None and ws_b is not None:
            
            def prep_siamese(crop):
                c = cv2.resize(crop, (224, 224))
                return torch.from_numpy(c).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb_a = self.siamese(prep_siamese(ws_a))
                emb_b = self.siamese(prep_siamese(ws_b))
                sim = torch.nn.functional.cosine_similarity(emb_a, emb_b).item()
            
            report["scores"]["identity_similarity"] = sim
            
            if sim > 0.92: 
                report["verdict"] = "FRAUD"
                report["flags"].append("DUPLICATE_VEHICLE_IDENTITY")

        return report


if __name__ == "__main__":
    orchestrator = ProductionFraudOrchestrator()

    result = orchestrator.analyze_claim("data/claim_01_a.jpg", "data/claim_01_b.jpg")
    
    print("-" * 30)
    print(f"ANALYSIS COMPLETE")
    print(f"VERDICT: {result['verdict']}")
    print(f"FLAGS: {', '.join(result['flags']) if result['flags'] else 'None'}")
    print(f"SCORES: {result['scores']}")
    print("-" * 30)