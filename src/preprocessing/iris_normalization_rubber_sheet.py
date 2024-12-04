import cv2
import numpy as np
import csv
import os
from tqdm import tqdm

def read_ellipse_data(csv_file):
    """Read all ellipse data from the CSV file."""
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

def crop_iris(image, iris_center, iris_axes, iris_angle):
    """Crop the iris region from the image based on the detected ellipse."""
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.ellipse(mask, (int(iris_center[0]), int(iris_center[1])), (int(iris_axes[0]), int(iris_axes[1])), iris_angle, 0, 360, 255, -1)
    cropped_iris = cv2.bitwise_and(image, image, mask=mask)
    return cropped_iris

def daugman_rubber_sheet_model(iris_image, iris_center, iris_axes, output_size=(64, 512)):
    """Apply Daugman's Rubber Sheet Model to transform the iris to a fixed size."""
    polar_iris = np.zeros(output_size, dtype=np.uint8)
    theta_range = np.linspace(0, 2 * np.pi, output_size[1])
    radius_range = np.linspace(0, iris_axes[0], output_size[0])

    for i, r in enumerate(radius_range):
        for j, theta in enumerate(theta_range):
            x = int(iris_center[0] + r * np.cos(theta))
            y = int(iris_center[1] + r * np.sin(theta))
            if 0 <= x < iris_image.shape[1] and 0 <= y < iris_image.shape[0]:
                polar_iris[i, j] = iris_image[y, x]
    
    return polar_iris

def resize_and_pad(image, target_size=(224, 224)):
    """Resize image while maintaining aspect ratio and pad to target size."""
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    
    pad_h = (target_size[0] - resized_image.shape[0]) // 2
    pad_w = (target_size[1] - resized_image.shape[1]) // 2
    
    padded_image = cv2.copyMakeBorder(resized_image, pad_h, target_size[0] - resized_image.shape[0] - pad_h, 
                                      pad_w, target_size[1] - resized_image.shape[1] - pad_w, 
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded_image

def process_and_save_image(row, output_base_dir):
    """Process a single image based on the row data and save it to the output directory."""
    image_path = row['Image Path']
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    pupil_center = (int(float(row['Pupil Center X'])), int(float(row['Pupil Center Y'])))
    iris_center = (float(row['Iris Center X']), float(row['Iris Center Y']))
    iris_axes = (float(row['Iris Axis Major']), float(row['Iris Axis Minor']))
    iris_angle = float(row['Iris Angle'])
    
    cropped_iris = crop_iris(image, iris_center, iris_axes, iris_angle)
    normalized_iris = daugman_rubber_sheet_model(cropped_iris, iris_center, iris_axes)
    final_image = resize_and_pad(normalized_iris, target_size=(224, 224))
    
    # Modify the path for saving the processed image
    output_path = image_path.replace('/home/hous/Desktop/IRIS_IDentification/data/filtred_100', output_base_dir)
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    cv2.imwrite(output_path, final_image)
    print(f"Normalized and resized iris saved to {output_path}")

def main():
    csv_file = "/home/hous/Desktop/IRIS_IDentification/data/filtered_ellipse_data.csv"  # Path to the CSV file
    output_base_dir = "/home/hous/Desktop/IRIS_IDentification/data/processed_normalized_100"  # Base directory for saving processed images
    
    # Read all ellipse data
    data = read_ellipse_data(csv_file)
    
    # Process each image and save with the modified path
    for row in tqdm(data, desc="Processing Images", unit="image"):
        process_and_save_image(row, output_base_dir)

if __name__ == "__main__":
    main()