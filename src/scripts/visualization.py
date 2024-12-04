import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def plot_embeddings(embeddings, labels, save_path='embedding_plot.png'):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.savefig(save_path)
    plt.show()

def plot_roc_curve(fpr, tpr, save_path='roc_curve.png'):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    embeddings = np.load('/path/to/embeddings.npy')
    labels = np.load('/path/to/labels.npy')
    
    plot_embeddings(embeddings, labels)
    
    fpr = np.load('/path/to/fpr.npy')
    tpr = np.load('/path/to/tpr.npy')
    plot_roc_curve(fpr, tpr)