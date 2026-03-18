import torch
import torch.nn as nn
from torchvision import models

class SiameseNetwork(nn.Module):
    """Staff-level Siamese architecture for identity matching."""
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Using ResNet50 for high-dimensional feature extraction
        self.encoder = models.resnet50(weights='DEFAULT')
        self.encoder.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128) # Project to 128-d embedding space
        )

    def forward(self, x):
        return self.encoder(x)

    def get_similarity(self, emb1, emb2):
        return torch.nn.functional.cosine_similarity(emb1, emb2)
    


class CarPoseModel(nn.Module):
    def __init__(self, num_classes=8):
        super(CarPoseModel, self).init__()
       
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)    