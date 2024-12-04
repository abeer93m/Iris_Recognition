import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.resnet50_model import ResNet50Embedding
from sklearn.metrics import pairwise_distances
from triplet_dataset import TripletDataset

def validate_model(model, dataloader):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for anchor, positive, negative, labels in dataloader:
            anchor = anchor.cuda()
            anchor_output = model(anchor)
            all_embeddings.append(anchor_output.cpu())
            all_labels.append(labels)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate pairwise distances between all embeddings
    distances = pairwise_distances(all_embeddings.numpy(), metric='euclidean')
    
    return distances, all_labels

if __name__ == "__main__":
    model = ResNet50Embedding()
    model.load_state_dict(torch.load('/path/to/model_checkpoint.pth'))
    model = model.cuda()

    val_dataset = TripletDataset('/path/to/validation_triplets.txt', transform=transforms.ToTensor())
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    distances, labels = validate_model(model, val_dataloader)
    # Save or further process distances and labels as needed