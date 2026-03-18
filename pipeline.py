import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, SuperGlueForKeypointMatching

class FraudDetectionEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.segmenter = YOLO('yolo11n-seg.pt') # Lightweight and fast
        self.sg_proc = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
        self.sg_model = SuperGlueForKeypointMatching.from_pretrained("magic-leap-community/superglue_outdoor").to(self.device).eval()

    def segment_car_and_bg(self, frame):
        results = self.segmenter(frame, classes=[2], verbose=False)
        if not results[0].masks: return None, None
        mask = results[0].masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        car = cv2.bitwise_and(frame, frame, mask=mask_binary)
        bg = cv2.bitwise_and(frame, frame, mask=(1 - mask_binary))
        return car, bg

    def match_features(self, img1, img2, output_path="viz_match.jpg"):
        i1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        i2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        
        inputs = self.sg_proc(images=[[i1, i2]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sg_model(**inputs)
        
        sizes = [[(i1.height, i1.width), (i2.height, i2.width)]]
        raw = self.sg_proc.post_process_keypoint_matching(outputs, sizes)[0]
        
        matches = raw["matches"]
        self._visualize(img1, img2, raw["keypoints0"], raw["keypoints1"], matches, output_path)
        return (matches != -1).sum().item()

    def _visualize(self, img1, img2, kp0, kp1, matches, path):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1], canvas[:h2, w1:] = img1, img2
        for i, m in enumerate(matches):
            if m != -1:
                p1, p2 = (int(kp0[i][0]), int(kp0[i][1])), (int(kp1[m][0] + w1), int(kp1[m][1]))
                cv2.line(canvas, p1, p2, (0, 255, 0), 1)
        cv2.imwrite(path, canvas)