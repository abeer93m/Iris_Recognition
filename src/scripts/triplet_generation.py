import os
import random
from tqdm import tqdm

def load_preprocessed_images(base_path):
    images = {}
    for subject_folder in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_folder)
        if not os.path.isdir(subject_path):
            continue
        images[subject_folder] = {'R': [], 'L': []}
        for side in ['R', 'L']:
            side_folder = os.path.join(subject_path, side)
            if os.path.exists(side_folder):
                for img_name in os.listdir(side_folder):
                    img_path = os.path.join(side_folder, img_name)
                    images[subject_folder][side].append(img_path)
            else:
                print(f"Warning: {side_folder} does not exist")
    return images

def generate_triplet(subject, subject_images, all_images, num_triplets_per_subject=815):
    triplets = []
    
    # Get the available sides (either R, L, or both)
    available_sides = [side for side in ['R', 'L'] if len(subject_images[side]) > 1]
    
    if not available_sides:
        print(f"Warning: Not enough images to generate triplets for subject {subject}")
        return triplets
    
    for _ in range(num_triplets_per_subject):
        side = random.choice(available_sides)
        available_images = subject_images[side]
        
        # Select anchor and positive images
        anchor_img = random.choice(available_images)
        positive_img = random.choice([img for img in available_images if img != anchor_img])
        
        # Attempt to select a negative image
        negative_img = None
        attempts = 0
        while negative_img is None and attempts < 10:  # Try up to 10 times to find a valid negative image
            negative_subject = random.choice([subj for subj in all_images if subj != subject])
            negative_side = random.choice(['R', 'L'])
            if all_images[negative_subject][negative_side]:
                negative_img = random.choice(all_images[negative_subject][negative_side])
            attempts += 1
        
        if negative_img is None:
            print(f"Warning: Could not find a valid negative image for {subject} after 10 attempts.")
            continue

        triplets.append((anchor_img, positive_img, negative_img))
    
    return triplets

def save_triplets(triplets, save_path):
    with open(save_path, 'w') as f:
        for triplet in triplets:
            f.write(','.join(triplet) + '\n')

if __name__ == "__main__":
    base_path = 'D:/Iris Identification/IRIS_IDentification/data/processed_normalized_100'
    save_path = 'D:/Iris Identification/IRIS_IDentification/data/triplets.txt'
    num_triplets_per_subject = 815  # Approximate number to hit the target of 75,000 triplets
    
    # Set random seed for reproducibility
    random.seed(42)
    
    all_images = load_preprocessed_images(base_path)
    triplets = []
    for subject in tqdm(all_images, desc="Generating triplets"):
        subject_images = all_images[subject]
        subject_triplets = generate_triplet(subject, subject_images, all_images, num_triplets_per_subject)
        triplets.extend(subject_triplets)
    
    # Shuffle the triplets
    random.shuffle(triplets)
    
    save_triplets(triplets, save_path)
    print(f"Generated {len(triplets)} triplets in total.")