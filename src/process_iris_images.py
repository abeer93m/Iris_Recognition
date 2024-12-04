import os
import cv2
from preprocessing.mask_darkregion import detect_and_replace_black_borders
from pupil_localization import preprocess_image, compute_gradients, detect_candidate_points, create_border_mask, localize_pupil

def process_image(image_path, output_path, output_original_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        # Step 1: Detect and replace black borders
        corrected_image = detect_and_replace_black_borders(image, iterations=2)
        
        # Step 2: Preprocess for pupil detection
        preprocessed_image = preprocess_image(corrected_image)
        
        # Step 3: Compute gradients
        gradient_magnitude, gradient_direction = compute_gradients(preprocessed_image)
        
        # Step 4: Detect candidate points
        candidate_mask = detect_candidate_points(preprocessed_image, gradient_magnitude)
        
        # Step 5: Create border mask
        border_mask = create_border_mask(preprocessed_image.shape, border_width=min(preprocessed_image.shape)//8)
        
        # Step 6: Localize the pupil using elliptical fitting
        final_image, ellipse, hough_space = localize_pupil(preprocessed_image, candidate_mask, gradient_direction, border_mask)
        
        # Save the result image with the ellipse drawn on the preprocessed image
        cv2.imwrite(output_path, final_image)
        print(f"Processed and saved: {output_path}")

        # Step 7: Draw the ellipse on the original image
        if ellipse is not None:
            original_color_image = cv2.imread(image_path)
            cv2.ellipse(original_color_image, ellipse, (0, 0, 255), 2)
            cv2.imwrite(output_original_path, original_color_image)
            print(f"Original image with ellipse saved to: {output_original_path}")
        else:
            print(f"No ellipse was found for: {image_path}")
    else:
        print(f"Could not load image from {image_path}")

def process_images_in_folder(input_folder, output_folder, output_original_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_original_folder):
        os.makedirs(output_original_folder)
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(image_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                output_original_path = os.path.join(output_original_folder, relative_path)
                output_dir = os.path.dirname(output_path)
                output_original_dir = os.path.dirname(output_original_path)
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if not os.path.exists(output_original_dir):
                    os.makedirs(output_original_dir)
                
                process_image(image_path, output_path, output_original_path)

def main():
    input_folder = "D:/Iris Identification/IRIS_IDentification/data/filtered/00000/L"  # Replace with your folder path
    output_folder = "D:/Iris Identification/IRIS_IDentification/data/processed"  # Replace with your output folder path
    output_original_folder = "D:/Iris Identification/IRIS_IDentification/data/processed_original"  # Replace with your original image output folder path
    
    # Process all images in the input folder and save results to the output folder
    process_images_in_folder(input_folder, output_folder, output_original_folder)

if __name__ == "__main__":
    main()
