import os
import torch
import numpy as np
import sys
from check_ellipses import process_csv_file
from iris_localization import process_images_in_folder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from preprocessing.iris_normalization_rubber_sheet import process_and_save_image, read_ellipse_data
from preprocessing.mask_darkregion import detect_and_replace_black_borders
from resnet50_model import ResNet50Embedding, initialize_model
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import csv
import cv2

# Step 1: Remove Dark Regions
def remove_dark_regions(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, file)
                
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    processed_image = detect_and_replace_black_borders(image)
                    cv2.imwrite(output_path, processed_image)
                else:
                    print(f"Could not load image: {image_path}")

# Step 2: Iris Detection and Localization
def detect_and_process_iris(processed_folder, iris_output_folder, csv_file):
    process_images_in_folder(processed_folder, iris_output_folder, csv_file)

# Step 3: Filter Abnormally Large Irises
def filter_large_irises(input_csv, filtered_csv, log_file):
    process_csv_file(input_csv, filtered_csv, log_file)

# Step 4: Iris Normalization
def normalize_iris_images(filtered_csv, normalized_folder):
    data = read_ellipse_data(filtered_csv)
    for row in tqdm(data, desc="Normalizing Images", unit="image"):
        output_path = process_and_save_image(row, normalized_folder)
        print(f"Processed image saved to: {output_path}")

# Step 5: Inference
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

def infer_user_identity(model, normalized_folder, known_user_embeddings, threshold=0.2):
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    embeddings = []
    for root, _, files in os.walk(normalized_folder):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, file)
                print(f"Processing image for embedding: {image_path}")
                image_tensor = preprocess_image(image_path, transform)
                embedding = get_embedding(model, image_tensor)
                embeddings.append(embedding)

    if not embeddings:
        print("No embeddings were generated. Please check your data.")
        return
    
    # Average the embeddings
    avg_embedding = np.mean(embeddings, axis=0)

    # Compare the average embedding to the known user embeddings
    min_distance = float('inf')
    matched_user = None
    for user_id, known_embedding in known_user_embeddings.items():
        distance = np.linalg.norm(avg_embedding - known_embedding)
        if distance < min_distance:
            min_distance = distance 
            matched_user = user_id
    
    # Determine if the user is verified based on the threshold
    if min_distance < threshold:
        print(f"User is verified as User {matched_user} with distance {min_distance:.4f}")
    else:
        print(f"User is not verified. Closest match is User {matched_user} with distance {min_distance:.4f}")

def main():
    input_folder = "D:/Iris Identification/IRIS_Identification/data/inference/00008"  # Folder with new user images
    processed_folder = "D:/Iris Identification/IRIS_Identification/data/inference/processed"  # Folder for images after dark region removal
    iris_output_folder = "D:/Iris Identification/IRIS_Identification/data/inference/iris_output"  # Folder for processed images
    normalized_folder = "D:/Iris Identification/IRIS_Identification/data/inference/processed"  # Folder for normalized images
    csv_file = "D:/Iris Identification/IRIS_Identification/data/inference/ellipse_data.csv"  # CSV file for ellipse data
    filtered_csv = "D:/Iris Identification/IRIS_Identification/data/inference/filtered_ellipse_data.csv"  # CSV file for filtered ellipse data
    log_file = "D:/Iris Identification/IRIS_Identification/data/inference/log_large_iris_axes.csv"  # Log file for large iris axes

    checkpoint_path = "D:/Iris Identification/IRIS_Identification/checkpoints/best_model_epoch_16.pth"  # Trained model checkpoint
    known_user_embeddings_path = "D:/Iris Identification/IRIS_Identification/data/known_user_embeddings.npy"  # Precomputed embeddings of known users
    
    # Step 1: Remove dark regions
    print("Step 1: Removing dark regions...")
    remove_dark_regions(input_folder, processed_folder)
    
    # Step 2: Detect and localize the iris
    print("Step 2: Detecting and processing iris...")
    detect_and_process_iris(processed_folder, iris_output_folder, csv_file)
    
    # Step 3: Filter abnormally large irises
    print("Step 3: Filtering large irises...")
    filter_large_irises(csv_file, filtered_csv, log_file)
    
    # Step 4: Normalize the iris images
    print("Step 4: Normalizing iris images...")
    normalize_iris_images(filtered_csv, normalized_folder)
    
    # Step 5: Load the model
    print("Step 5: Loading the model...")
    model = load_model(checkpoint_path)
    
    # Step 6: Load known user embeddings
    print("Step 6: Loading known user embeddings...")
    known_user_embeddings = np.load(known_user_embeddings_path, allow_pickle=True).item()

    # Step 7: Inference
    print("Step 7: Performing inference...")
    infer_user_identity(model, normalized_folder, known_user_embeddings)

if __name__ == "__main__":
    main()