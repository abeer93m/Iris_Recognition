import os
import cv2
import numpy as np
import shutil

def load_and_filter_images(image_dir, output_dir):
    for subject in os.listdir(image_dir):
        subject_path = os.path.join(image_dir, subject)
        if os.path.isdir(subject_path):
            for eye in ['L', 'R']:
                eye_path = os.path.join(subject_path, eye)
                if os.path.exists(eye_path):
                    for img_name in os.listdir(eye_path):
                        img_path = os.path.join(eye_path, img_name)
                        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            if is_eye_open(image):  # Check if the eye is open
                                save_image(subject, eye, img_name, image, output_dir)
                            else:
                                print(f"Discarding closed-eye image: {img_path}")

def is_eye_open(image, threshold=0.3):
    """
    A basic method to check if the eye is open.
    - threshold: A parameter to fine-tune the sensitivity.
    Returns True if the eye is open, otherwise False.
    """
    mean_intensity = np.mean(image)
    return mean_intensity > threshold

def save_image(subject, eye, img_name, image, output_dir):
    subject_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_dir, exist_ok=True)
    eye_dir = os.path.join(subject_dir, eye)
    os.makedirs(eye_dir, exist_ok=True)
    img_path = os.path.join(eye_dir, img_name)
    cv2.imwrite(img_path, image)

def main():
    input_dir = "data/raw"
    output_dir = "data/filtered"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Clear the filtered folder if it exists
    
    load_and_filter_images(input_dir, output_dir)
    print(f"Filtering completed. Open-eye images saved to {output_dir}")

if __name__ == "__main__":
    main()
