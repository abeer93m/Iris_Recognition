import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import transforms
from resnet50_model import ResNet50Embedding, initialize_model
from PIL import Image
from tqdm import tqdm

def load_model(checkpoint_path, embedding_dim=128):
    model, _, _ = initialize_model(embedding_dim=embedding_dim, margin=1.0)
    # Check if CUDA is available and set the appropriate map_location
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint_path))
        model = model.cuda()
    else:
        # Map tensors to CPU
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def get_embedding(model, image_tensor):
    if torch.cuda.is_available():
        image_tensor = image_tensor.unsqueeze(0).cuda()
    else:
        image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.cpu().numpy().flatten()

def compute_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def compute_metrics(distances, labels):
    # Invert distances
    scores = -distances  # Higher scores indicate more similarity

    # Compute ROC metrics
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate EER
    fnr = 1 - tpr
    abs_diffs = np.abs(fnr - fpr)
    idx_eer = np.nanargmin(abs_diffs)
    eer = fpr[idx_eer]
    eer_threshold = thresholds[idx_eer]
    
    # Compute optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Apply threshold to get predictions
    predictions = scores >= optimal_threshold
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    # FAR and FRR at optimal threshold
    far = fpr[optimal_idx]
    frr = fnr[optimal_idx]
    
    return accuracy, precision, recall, f1, roc_auc, eer, far, frr, optimal_threshold, eer_threshold, fpr, tpr, thresholds, predictions

def plot_roc_curve(fpr, tpr, roc_auc, save_path='roc_curve.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")

def plot_distance_distribution(distances, labels, save_path='distance_distribution.png'):
    plt.figure(figsize=(8, 6))
    sns.histplot(distances[labels == 1], color='green', label='Genuine Pairs', kde=True, stat="density", bins=50, alpha=0.6)
    sns.histplot(distances[labels == 0], color='red', label='Impostor Pairs', kde=True, stat="density", bins=50, alpha=0.6)
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.title('Distribution of Distances for Genuine and Impostor Pairs')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Distance distribution plot saved to {save_path}")

def plot_confusion_matrix(labels, predictions, save_path='confusion_matrix.png'):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Impostor', 'Genuine'], yticklabels=['Impostor', 'Genuine'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_det_curve(fpr, fnr, save_path='det_curve.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, fnr, color='purple', lw=2)
    plt.plot([0, 1], [1, 0], color='grey', lw=1, linestyle='--')
    plt.xlabel('False Accept Rate (FAR)')
    plt.ylabel('False Reject Rate (FRR)')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"DET curve saved to {save_path}")

def plot_eer(fpr, tpr, thresholds, eer, eer_threshold, save_path='eer_plot.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.scatter(fpr[np.argmin(np.abs(fpr - eer))], tpr[np.argmin(np.abs(tpr - (1 - eer)))],
                marker='o', color='red', label=f'EER = {eer:.4f}')
    plt.text(fpr[np.argmin(np.abs(fpr - eer))] + 0.02, tpr[np.argmin(np.abs(tpr - (1 - eer)))],
             f'Threshold: {eer_threshold:.4f}', color='red')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve with EER Point')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"EER plot saved to {save_path}")

def main():
    # Paths
    checkpoint_path = 'D:/Iris Identification/IRIS_IDentification/checkpoints/best_model_epoch_16.pth'  # Update with correct path
    test_pairs_file = 'D:/Iris Identification/IRIS_IDentification/data/test_pairs.txt'  # Update with correct path
    output_dir = 'D:/Iris Identification/IRIS_IDentification/results/'  # Update with desired output directory
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = load_model(checkpoint_path, embedding_dim=128)
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Assuming RGB images
    ])
    
    distances = []
    labels = []
    
    # Read test pairs
    with open(test_pairs_file, 'r') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="Processing pairs"):
        image1_path, image2_path, label = line.strip().split(',')
        label = int(label)
        
        # Preprocess images
        image1 = preprocess_image(image1_path, transform)
        image2 = preprocess_image(image2_path, transform)
        
        # Get embeddings
        embedding1 = get_embedding(model, image1)
        embedding2 = get_embedding(model, image2)
        
        # Compute distance between embeddings
        distance = compute_distance(embedding1, embedding2)
        distances.append(distance)
        labels.append(label)
    
    distances = np.array(distances)
    labels = np.array(labels)
    
    # Compute metrics
    (accuracy, precision, recall, f1, roc_auc, eer, far, frr,
     optimal_threshold, eer_threshold, fpr, tpr, thresholds, predictions) = compute_metrics(distances, labels)
    
    # Print Evaluation Metrics
    print(f"\nEvaluation Metrics:")
    print(f"-------------------")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print(f"ROC AUC        : {roc_auc:.4f}")
    print(f"EER            : {eer:.4f} at threshold {eer_threshold:.4f}")
    print(f"Optimal Threshold : {optimal_threshold:.4f}")
    print(f"FAR            : {far:.4f} at optimal threshold")
    print(f"FRR            : {frr:.4f} at optimal threshold")
    
    # Save metrics to a text file
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Evaluation Metrics:\n")
        f.write("-------------------\n")
        f.write(f"Accuracy       : {accuracy:.4f}\n")
        f.write(f"Precision      : {precision:.4f}\n")
        f.write(f"Recall         : {recall:.4f}\n")
        f.write(f"F1-Score       : {f1:.4f}\n")
        f.write(f"ROC AUC        : {roc_auc:.4f}\n")
        f.write(f"EER            : {eer:.4f} at threshold {eer_threshold:.4f}\n")
        f.write(f"Optimal Threshold : {optimal_threshold:.4f}\n")
        f.write(f"FAR            : {far:.4f} at optimal threshold\n")
        f.write(f"FRR            : {frr:.4f} at optimal threshold\n")
    print(f"Metrics saved to {metrics_path}")
    
    # Create ROC Curve plot
    roc_curve_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(fpr, tpr, roc_auc, save_path=roc_curve_path)
    
    # Create Distance Distribution plot
    distance_dist_path = os.path.join(output_dir, 'distance_distribution.png')
    plot_distance_distribution(distances, labels, save_path=distance_dist_path)
    
    # Create Confusion Matrix plot
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(labels, predictions, save_path=confusion_matrix_path)
    
    # Optional: Create DET Curve plot
    det_curve_path = os.path.join(output_dir, 'det_curve.png')
    plot_det_curve(fpr, 1 - tpr, save_path=det_curve_path)
    
    # Optional: Create EER Plot
    eer_plot_path = os.path.join(output_dir, 'eer_plot.png')
    plot_eer(fpr, tpr, thresholds, eer, eer_threshold, save_path=eer_plot_path)

if __name__ == "__main__":
    main()
