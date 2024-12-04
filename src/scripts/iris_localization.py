import cv2
import os
import numpy as np
import csv
import concurrent.futures
from tqdm import tqdm
from pupil_localization import preprocess_image, compute_gradients, detect_candidate_points, refine_radius, fit_ellipse_to_boundary, create_border_mask, localize_pupil, search_boundary

def localize_iris(image, pupil_center, pupil_axes, pupil_angle, gradient_direction, border_mask, scale_factor=2.5, max_radius_increase=50):
    """Localize the iris by expanding the region around the detected pupil."""
    directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    
    initial_iris_axes = (int(pupil_axes[0] * scale_factor), int(pupil_axes[1] * scale_factor))
    initial_iris_ellipse = (pupil_center, initial_iris_axes, pupil_angle)

    refined_iris_radius = max(initial_iris_axes) + max_radius_increase
    iris_boundary_points = search_boundary(image, pupil_center, refined_iris_radius, directions)
    
    iris_image, iris_ellipse = fit_ellipse_to_boundary(image, iris_boundary_points)

    if iris_ellipse:
        long_axis, short_axis = iris_ellipse[1]
        refined_iris_axes = (long_axis * 0.65, int(short_axis * 0.5))
        refined_iris_ellipse = (iris_ellipse[0], refined_iris_axes, iris_ellipse[2])
    else:
        refined_iris_ellipse = iris_ellipse

    return iris_image, refined_iris_ellipse

def detect_pupil_and_iris(image_info):
    image_path, output_path, csv_writer = image_info
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            preprocessed_image = preprocess_image(image)
            gradient_magnitude, gradient_direction = compute_gradients(preprocessed_image)
            candidate_mask = detect_candidate_points(preprocessed_image, gradient_magnitude)
            border_mask = create_border_mask(preprocessed_image.shape, border_width=min(preprocessed_image.shape)//8)
            final_pupil_image, pupil_ellipse, hough_space = localize_pupil(preprocessed_image, candidate_mask, gradient_direction, border_mask)
            
            if pupil_ellipse is not None:
                pupil_center = (int(pupil_ellipse[0][0]), int(pupil_ellipse[0][1]))
                pupil_axes = (int(pupil_ellipse[1][0] / 2), int(pupil_ellipse[1][1] / 2))
                pupil_angle = pupil_ellipse[2]
                
                iris_image, iris_ellipse = localize_iris(preprocessed_image, pupil_center, pupil_axes, pupil_angle, gradient_direction, border_mask)
                
                result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.ellipse(result_image, pupil_ellipse, (0, 0, 255), 2)
                if iris_ellipse is not None:
                    cv2.ellipse(result_image, iris_ellipse, (0, 255, 0), 2)
                
                cv2.imwrite(output_path, result_image)
                
                if iris_ellipse and iris_ellipse[1][0] > 0 and iris_ellipse[1][1] > 0:
                    csv_writer.writerow([image_path, *pupil_center, *pupil_axes, pupil_angle, 
                                         iris_ellipse[0][0], iris_ellipse[0][1], 
                                         iris_ellipse[1][0] / 2, iris_ellipse[1][1] / 2, iris_ellipse[2]])
            else:
                print(f"Pupil could not be detected in {image_path}")
        else:
            print(f"Could not load image from {image_path}")
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_images_in_folder(input_folder, output_folder, csv_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_infos = []
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Pupil Center X", "Pupil Center Y", "Pupil Axis Major", "Pupil Axis Minor", "Pupil Angle", 
                         "Iris Center X", "Iris Center Y", "Iris Axis Major", "Iris Axis Minor", "Iris Angle"])
        
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(image_path, input_folder)
                    output_path = os.path.join(output_folder, relative_path)
                    output_dir = os.path.dirname(output_path)
                    
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    image_infos.append((image_path, output_path, writer))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(detect_pupil_and_iris, image_infos), total=len(image_infos), desc="Processing Images"))

def main():
    input_folder = "/home/hous/Desktop/IRIS_IDentification/data/filtred_100"  # Input folder containing raw images
    output_folder = "/home/hous/Desktop/IRIS_IDentification/data/processed_100"  # Output folder for processed images
    csv_file = "/home/hous/Desktop/IRIS_IDentification/data/ellipse_data_100.csv"  # CSV file to save ellipse data
    
    process_images_in_folder(input_folder, output_folder, csv_file)

if __name__ == "__main__":
    main()
