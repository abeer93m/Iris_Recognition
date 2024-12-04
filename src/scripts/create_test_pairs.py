import os
import random

def get_image_paths(data_dir):
    """ 
    Retrieve all image paths organized by subject and side (R or L).
    The data_dir is expected to contain subdirectories for each subject, 
    each with 'R' and 'L' subdirectories for right and left eye images.
    """
    subject_dict = {}
    for subject in os.listdir(data_dir):
        subject_dir = os.path.join(data_dir, subject)
        if os.path.isdir(subject_dir):
            subject_dict[subject] = {'R': [], 'L': []}
            for side in ['R', 'L']:
                side_dir = os.path.join(subject_dir, side)
                if os.path.isdir(side_dir):
                    for image_name in os.listdir(side_dir):
                        image_path = os.path.join(side_dir, image_name)
                        subject_dict[subject][side].append(image_path)
    return subject_dict

def generate_pairs(subject_dict, num_pairs=100):
    """ 
    Generate pairs of images and labels. 
    Same class pairs are labeled as 1, and different class pairs as 0.
    """
    pairs = []
    subjects = list(subject_dict.keys())

    # Generate same class pairs (label 1)
    for subject, images_by_side in subject_dict.items():
        for side in ['R', 'L']:
            images = images_by_side[side]
            if len(images) > 1:
                for _ in range(min(num_pairs // 4, len(images) * (len(images) - 1) // 2)):
                    img1, img2 = random.sample(images, 2)
                    pairs.append((img1, img2, 1))
    
    # Generate different class pairs (label 0)
    for _ in range(num_pairs // 2):
        subject1, subject2 = random.sample(subjects, 2)
        side1 = random.choice(['R', 'L'])
        side2 = random.choice(['R', 'L'])
        
        if subject_dict[subject1][side1] and subject_dict[subject2][side2]:
            img1 = random.choice(subject_dict[subject1][side1])
            img2 = random.choice(subject_dict[subject2][side2])
            pairs.append((img1, img2, 0))
    
    return pairs

def save_pairs(pairs, output_file):
    """ 
    Save the generated pairs to a file in the format: image1_path, image2_path, label.
    """
    with open(output_file, 'w') as f:
        for img1, img2, label in pairs:
            f.write(f"{img1},{img2},{label}\n")

if __name__ == "__main__":
    # Parameters
    data_dir = 'D:/Iris Identification/IRIS_IDentification/data/processed_normalized_100'  # Update with your path
    output_file = 'D:/Iris Identification/IRIS_IDentification/data/test_pairs.txt'   # Update with your path
    num_pairs = 300  # Number of pairs to generate, adjust as needed
    
    # Generate image paths organized by subject and side
    subject_dict = get_image_paths(data_dir)
    
    # Debugging: Check the number of subjects
    print(f"Number of subjects found: {len(subject_dict)}")
    print(f"Subjects: {list(subject_dict.keys())}")   
    # Generate pairs
    pairs = generate_pairs(subject_dict, num_pairs=num_pairs)
    
    # Save pairs to file
    save_pairs(pairs, output_file)
    
    print(f"Generated {len(pairs)} pairs and saved to {output_file}")