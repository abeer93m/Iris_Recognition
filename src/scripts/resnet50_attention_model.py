# resnet50_attention_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B x C x N
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = F.softmax(energy, dim=-1)  # B x N x N
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out

class ResNet50AttentionEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ResNet50AttentionEmbedding, self).__init__()
        # Load the pre-trained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=True)
        self.attention = SelfAttention(in_dim=self.resnet50.layer4[2].conv3.out_channels)  # Add attention after the last conv layer of ResNet50
        
        # Replace the final fully connected layer
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, embedding_dim)
    
    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        
        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        # Apply the attention mechanism
        x = self.attention(x)
        
        # Global Average Pooling
        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)

        # Final fully connected layer
        x = self.resnet50.fc(x)

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

def initialize_attention_model(embedding_dim=128, margin=1.0):
    model = ResNet50AttentionEmbedding(embedding_dim=embedding_dim)
    criterion = TripletLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model, criterion, optimizer

if __name__ == "__main__":
    # Example usage
    model, criterion, optimizer = initialize_attention_model()
    print(model)