# training_with_attention.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch.cuda.amp as amp
from torchvision import transforms
from resnet50_attention_model import ResNet50AttentionEmbedding, TripletLoss, initialize_attention_model
import wandb
from PIL import Image
import numpy as np

# Initialize wandb
wandb.init(project="iris-identification-attention", config={
    "epochs": 25,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "k_folds": 5,
    "embedding_dim": 128,
    "margin": 1.0,
    "accumulation_steps": 4  # Number of mini-batches to accumulate before updating gradients
})

class TripletDataset(Dataset):
    def __init__(self, triplet_file, transform=None):
        self.triplets = []
        with open(triplet_file, 'r') as f:
            for line in f:
                anchor, positive, negative = line.strip().split(',')
                self.triplets.append((anchor, positive, negative))
        self.transform = transform
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]
        anchor = self.load_image(anchor_path)
        positive = self.load_image(positive_path)
        negative = self.load_image(negative_path)
        
        return anchor, positive, negative
    
    def load_image(self, path):
        image = Image.open(path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image

def evaluate_model(model, dataloader):
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()

            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)
            
            all_embeddings.extend([anchor_output.cpu().numpy(), positive_output.cpu().numpy(), negative_output.cpu().numpy()])
            all_labels.extend([0, 1, 1])  # Assuming 0 for genuine, 1 for impostor

    # Convert lists to numpy arrays
    embeddings = np.array(all_embeddings)
    labels = np.array(all_labels)
    
    # Calculate pairwise distances
    pos_dist = np.linalg.norm(embeddings[0] - embeddings[1], axis=1)
    neg_dist = np.linalg.norm(embeddings[0] - embeddings[2], axis=1)
    
    distances = np.concatenate([pos_dist, neg_dist])
    labels = np.concatenate([np.zeros_like(pos_dist), np.ones_like(neg_dist)])

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, distances)
    roc_auc = auc(fpr, tpr)
    
    # Calculate EER
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    print(f"EER: {eer:.4f}, AUC: {roc_auc:.4f}")
    
    return eer, roc_auc

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_size, num_epochs=25, save_dir='./checkpoints', accumulation_steps=4):
    best_loss = float('inf')
    scaler = amp.GradScaler()  # Initialize GradScaler for mixed precision training
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            optimizer.zero_grad()  # Reset gradients at the start of each epoch

            # Iterate over data
            for i, (anchor, positive, negative) in enumerate(dataloaders[phase]):
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
                
                with torch.set_grad_enabled(phase == 'train'):
                    with amp.autocast():  # Enable mixed precision for this block
                        anchor_output = model(anchor)
                        positive_output = model(positive)
                        negative_output = model(negative)
                        
                        loss = criterion(anchor_output, positive_output, negative_output)
                        loss = loss / accumulation_steps  # Normalize loss for accumulation
                    
                    if phase == 'train':
                        scaler.scale(loss).backward()  # Scale the loss and call backward()
                        
                        # Gradient accumulation step
                        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloaders[phase]):
                            scaler.step(optimizer)  # Unscales gradients and calls optimizer.step()
                            scaler.update()  # Updates the scale for next iteration
                            optimizer.zero_grad()
                
                running_loss += loss.item() * accumulation_steps  # Multiply by accumulation_steps to get the actual loss

                # Log the batch loss to wandb
                wandb.log({f"{phase}_batch_loss": loss.item() * accumulation_steps, "batch": i + epoch * len(dataloaders[phase])})
            
            epoch_loss = running_loss / dataset_size[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Log the epoch loss to wandb
            wandb.log({f"{phase}_loss": epoch_loss, "epoch": epoch})
            
            if phase == 'train':
                scheduler.step(epoch_loss)
            
            # Save the model if the validation loss improves
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_epoch_{epoch}.pth'))
                print(f'Model saved with loss: {best_loss:.4f}')
    
    return model

if __name__ == "__main__":
    # Parameters
    data_dir = '/home/hous/Desktop/IRIS_IDentification/data/processed_normalized_100'
    triplet_file = '/home/hous/Desktop/IRIS_IDentification/data/triplets.txt'
    config = wandb.config
    
    # Define image transformations with augmentation
    train_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.GaussianBlur(kernel_size=3)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Initialize model with attention, criterion, optimizer, and scheduler
    model, criterion, optimizer = initialize_attention_model(embedding_dim=config.embedding_dim, margin=config.margin)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    model = model.cuda()
    criterion = criterion.cuda()
    
    # Load dataset and create DataLoader
    dataset = TripletDataset(triplet_file, transform=train_transforms)
    
    kfold = KFold(n_splits=config.k_folds, shuffle=True)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{config.k_folds}')
        print('--------------------------')
        
        # Sample elements randomly from a given list of indices for train and validation
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        # Create data loaders
        dataloaders = {
            'train': DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=4),
            'val': DataLoader(dataset, batch_size=config.batch_size, sampler=val_sampler, num_workers=4)
        }
        
        dataset_size = {'train': len(train_idx), 'val': len(val_idx)}
        
        # Train model
        model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_size, num_epochs=config.epochs, accumulation_steps=config.accumulation_steps)
        
        # Evaluate model on validation set
        val_eer, val_auc = evaluate_model(model, dataloaders['val'])
        wandb.log({'val_eer': val_eer, 'val_auc': val_auc})
        
        print()

    # Finish wandb run
    wandb.finish()