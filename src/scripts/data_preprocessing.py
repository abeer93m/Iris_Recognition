import os
import cv2

def load_images(subject_path):
    images = {'R': [], 'L': []}
    for side in ['R', 'L']:
        folder_path = os.path.join(subject_path, side)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            images[side].append((img_name, image))
    return images

def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    return normalized_image

def save_preprocessed_image(image, save_path):
    cv2.imwrite(save_path, image * 255)

def preprocess_and_save(subject_path, save_base_path):
    images = load_images(subject_path)
    subject_id = os.path.basename(subject_path)
    for side, imgs in images.items():
        save_dir = os.path.join(save_base_path, subject_id, side)
        os.makedirs(save_dir, exist_ok=True)
        for img_name, image in imgs:
            preprocessed_image = preprocess_image(image)
            save_path = os.path.join(save_dir, img_name)
            save_preprocessed_image(preprocessed_image, save_path)

if __name__ == "__main__":
    data_dir = 'D:/Iris Identification/IRIS_Identification/raw data/MetaIris/S1'
    save_dir = 'D:/Iris Identification/IRIS_Identification/data/processed'
    for subject_folder in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject_folder)
        preprocess_and_save(subject_path, save_dir)