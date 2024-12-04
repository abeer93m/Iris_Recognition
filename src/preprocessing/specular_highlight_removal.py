import cv2
import numpy as np

def specular_highlight_removal(image, threshold_value, kernel_size, inpaint_radius, inpaint_method=cv2.INPAINT_TELEA):
    """
    Detect and remove specular highlights from an iris image caused by LED reflections.
    
    Parameters:
    - image: Grayscale image of the eye.
    - threshold_value: Threshold value to detect bright spots.
    - kernel_size: Size of the morphological kernel.
    - inpaint_radius: Radius for inpainting the highlights.
    - inpaint_method: Method to use for inpainting (TELEA or NS).
    
    Returns:
    - result_img: Image with specular highlights removed and darkened.
    - mask: The mask created during thresholding.
    """
    _, mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_mask = np.zeros_like(mask_opened)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 <= area <= 3000:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    
    # Expand the mask slightly more before inpainting
    expanded_mask = cv2.dilate(final_mask, kernel, iterations=3)
    
    # Perform inpainting with the specified method
    inpainted_img = cv2.inpaint(image, expanded_mask, inpaintRadius=inpaint_radius, flags=inpaint_method)
    
    # Darken the inpainted region slightly
    darkened_img = inpainted_img.copy()
    darkened_img[expanded_mask == 255] = darkened_img[expanded_mask == 255] * 0.7  # Reduce brightness by 30%
    
    return darkened_img, expanded_mask

def main():
    input_image_path = "/home/hous/Desktop/IRIS_IDentification/00000_3_R_04.jpg"  # Replace with your image path
    output_image_path = "processed_larger_darker.jpg"  # Replace with your output image path
    output_mask_path = "mask_larger_darker.jpg"  # Replace with your output mask path
    
    # Load the image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        # Use the selected configuration
        threshold_value = 230
        kernel_size = 5
        inpaint_radius = 10  # Increased inpainting radius
        inpaint_method = cv2.INPAINT_NS  # Continue using the 'NS' inpainting method
        
        # Process the image
        result_img, mask = specular_highlight_removal(image, threshold_value, kernel_size, inpaint_radius, inpaint_method)
        
        # Save the processed image and mask
        cv2.imwrite(output_image_path, result_img)
        cv2.imwrite(output_mask_path, mask)
        
        print(f"Processed image saved to {output_image_path}")
        print(f"Mask image saved to {output_mask_path}")
    else:
        print(f"Could not load image from {input_image_path}")

if __name__ == "__main__":
    main()