import cv2
import numpy as np

def detect_and_replace_black_borders(image, threshold=10, iterations=2):
    """
    Detect black borders and corners in the image and replace them with the average of light pixels in the image.
    Further, replace remaining black pixels with nearest light pixel value and smooth the frontier.
    
    Parameters:
    - image: Grayscale image.
    - threshold: Intensity threshold to consider a pixel as black.
    - iterations: Number of iterations to apply the black pixel replacement.
    
    Returns:
    - result_img: Image with black borders and corners replaced and smoothed.
    """
    h, w = image.shape
    
    black_mask = image < threshold
    light_pixels = image[~black_mask]
    average_light_value = int(np.mean(light_pixels)) if light_pixels.size > 0 else 255
    result_img = image.copy()
    
    left_region, right_region = w // 4, w - w // 4
    top_region, bottom_region = h // 8, h - h // 8

    for i in range(h):
        for j in range(left_region):
            if black_mask[i, j]:
                result_img[i, j] = average_light_value
        for j in range(right_region, w):
            if black_mask[i, j]:
                result_img[i, j] = average_light_value

    for i in range(top_region):
        for j in range(w):
            if black_mask[i, j]:
                result_img[i, j] = average_light_value
    for i in range(bottom_region, h):
        for j in range(w):
            if black_mask[i, j]:
                result_img[i, j] = average_light_value
    
    for _ in range(iterations):
        black_mask = result_img < threshold
        for i in range(h):
            for j in range(w):
                if black_mask[i, j]:
                    neighbors = []
                    if i > 0: neighbors.append(result_img[i-1, j])
                    if i < h-1: neighbors.append(result_img[i+1, j])
                    if j > 0: neighbors.append(result_img[i, j-1])
                    if j < w-1: neighbors.append(result_img[i, j+1])
                    if neighbors:
                        result_img[i, j] = int(np.mean(neighbors))

    result_img = cv2.GaussianBlur(result_img, (5, 5), sigmaX=1, sigmaY=1)

    return result_img