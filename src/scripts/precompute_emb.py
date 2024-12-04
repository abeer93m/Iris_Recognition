import os
import torch
import numpy as np
from resnet50_model import ResNet50Embedding, initialize_model
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def load_model(checkpoint_path, embedding_dim=128):
    model, _, _ = initialize_model(embedding_dim=embedding_dim, margin=1.0)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.cuda()
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.cuda()

def get_embedding(model, image_tensor):
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.cpu().numpy()

def compute_user_embeddings(data_dir, model, transform):
    user_embeddings = {}

    for user_folder in tqdm(os.listdir(data_dir), desc="Processing Users"):
        user_path = os.path.join(data_dir, user_folder)
        if not os.path.isdir(user_path):
            continue

        embeddings = []

        for side in ['R', 'L']:
            side_folder = os.path.join(user_path, side)
            if not os.path.exists(side_folder):
                continue

            for img_name in os.listdir(side_folder):
                img_path = os.path.join(side_folder, img_name)
                if img_name.endswith((".jpg", ".png", ".jpeg")):
                    image_tensor = preprocess_image(img_path, transform)
                    embedding = get_embedding(model, image_tensor)
                    embeddings.append(embedding)
        
        if embeddings:
            # Compute the average embedding for the user
            user_embeddings[user_folder] = np.mean(embeddings, axis=0)

    return user_embeddings

def save_embeddings(embeddings, output_path):
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")

def main():
    data_dir = "D:/Iris Identification/IRIS_Identification/data/processed_normalized_100"  # Directory with user folders containing normalized iris images
    checkpoint_path = "D:/Iris Identification/IRIS_Identification/checkpoints/best_model_epoch_16.pth"  # Trained model checkpoint
    output_path = "D:/Iris Identification/IRIS_Identification/data/known_user_embeddings.npy"  # Path to save the computed embeddings

    # Load the trained model
    model = load_model(checkpoint_path)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Compute embeddings for each user
    user_embeddings = compute_user_embeddings(data_dir, model, transform)

    # Save the embeddings to a file
    save_embeddings(user_embeddings, output_path)

if __name__ == "__main__":
    main()