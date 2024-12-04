import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

class ResNet50Embedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ResNet50Embedding, self).__init__()
        # Load the pre-trained ResNet50 model
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Replace the final fully connected layer
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, embedding_dim)
    
    def forward(self, x):
        x = self.resnet50(x)
        # L2 normalize the embeddings
        x = F.normalize(x, p=2, dim=1)
        return x

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

def initialize_model(embedding_dim=128, margin=1.0):
    model = ResNet50Embedding(embedding_dim=embedding_dim)
    criterion = TripletLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model, criterion, optimizer

if __name__ == "__main__":
    # Example usage
    model, criterion, optimizer = initialize_model()
    print(model)