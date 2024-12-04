import cv2
import numpy as np

def contrast_enhancement(image):
    """
    Enhance the contrast of the image to make features like eyelashes more distinct.
    
    Parameters:
    - image: Grayscale image of the eye.
    
    Returns:
    - enhanced_img: Contrast-enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(image)
    return enhanced_img

def eyelash_mitigation(image, kernel_size):
    """
    Mitigate the impact of eyelashes on iris boundary localization using a single morphological operation.
    
    Parameters:
    - image: Grayscale image of the eye.
    - kernel_size: Size of the kernel for the morphological operation.
    
    Returns:
    - mitigated_img: Image with reduced influence of eyelashes.
    """
    # Step 1: Contrast enhancement
    enhanced_img = contrast_enhancement(image)
    
    # Step 2: Morphological operation to remove or thin dark vertical lines (eyelashes)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    
    # Apply morphological open operation with horizontal kernel
    mitigated_img = cv2.morphologyEx(enhanced_img, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Step 3: Combine the mitigated image with the original using weighted sum
    blended_img = cv2.addWeighted(image, 0.8, mitigated_img, 0.2, 0)
    
    return blended_img

def main():
    input_image_path = "/mnt/data/00003_0_L_00.jpg"  # Replace with your image path
    output_image_path = "/mnt/data/eyelash_mitigated_refined.jpg"  # Replace with your output image path
    
    # Load the image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        # Test with a specific kernel size
        kernel_size = 7
        
        mitigated_img = eyelash_mitigation(image, kernel_size)
        
        # Save the mitigated image
        cv2.imwrite(output_image_path, mitigated_img)
        print(f"Eyelash mitigated image saved to {output_image_path}")
    else:
        print(f"Could not load image from {input_image_path}")

if __name__ == "__main__":
    main()