import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_gradients(image):
    """
    Compute the gradient magnitudes and directions of an image using the Sobel operator.
    
    Parameters:
    - image: Grayscale image of the eye.
    
    Returns:
    - gradient_magnitude: The magnitude of the gradient at each pixel.
    - gradient_direction: The direction of the gradient at each pixel.
    """
    # Compute the gradients along the x and y axes using Sobel operator
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=7)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=7)
    
    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Compute the gradient direction
    gradient_direction = np.arctan2(sobel_y, sobel_x)
    
    return gradient_magnitude, gradient_direction

def main():
    input_image_path = "/mnt/data/image.png"  # Replace with your image path
    output_magnitude_path = "/mnt/data/gradient_magnitude.png"
    output_direction_path = "/mnt/data/gradient_direction.png"
    
    # Load the image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        # Compute the gradients
        gradient_magnitude, gradient_direction = compute_gradients(image)
        
        # Save the gradient magnitude image
        plt.imsave(output_magnitude_path, gradient_magnitude, cmap='gray')
        
        # Normalize the direction for visualization (optional)
        gradient_direction_normalized = (gradient_direction + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        plt.imsave(output_direction_path, gradient_direction_normalized, cmap='hsv')  # HSV colormap to visualize angles
        
        print(f"Gradient magnitude saved to {output_magnitude_path}")
        print(f"Gradient direction saved to {output_direction_path}")
    else:
        print(f"Could not load image from {input_image_path}")

if __name__ == "__main__":
    main()
