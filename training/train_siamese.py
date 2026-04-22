import torch
import torch.nn as nn
from models import SiameseNetwork

def train_siamese(dataloader, epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiameseNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Triplet loss focuses on relative distance (Anchor, Positive, Negative)
    criterion = nn.TripletMarginLoss(margin=1.0)

    for epoch in range(epochs):
        for anchor, pos, neg in dataloader:
            optimizer.zero_grad()
            a_emb, p_emb, n_emb = model(anchor), model(pos), model(neg)
            loss = criterion(a_emb, p_emb, n_emb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "siamese_identity.pth")