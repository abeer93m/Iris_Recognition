import cv2
import numpy as np

def preprocess_image(image):
    """Preprocess the image to enhance features and reduce noise."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(image)
    blurred_img = cv2.GaussianBlur(enhanced_img, (7, 7), 0)
    return blurred_img

def compute_gradients(image):
    """Compute the gradient magnitude and direction of the image."""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=7)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_direction = np.arctan2(sobely, sobelx)
    return gradient_magnitude, gradient_direction

def detect_candidate_points(image, gradient_magnitude, threshold_intensity=50, threshold_gradient=50):
    """Detect candidate boundary points based on intensity and gradient thresholds."""
    candidate_mask = np.zeros_like(image, dtype=np.uint8)
    candidate_mask[(image < threshold_intensity) & (gradient_magnitude > threshold_gradient)] = 255
    return candidate_mask

def create_border_mask(image_shape, border_width=20):
    """Create a mask that penalizes points near the image border."""
    mask = np.ones(image_shape, dtype=np.float32)
    h, w = image_shape
    mask[border_width:h-border_width, border_width:w-border_width] = 1.0
    mask[:border_width, :] = np.linspace(0.2, 1.0, border_width)[:, None]
    mask[-border_width:, :] = np.linspace(1.0, 0.2, border_width)[:, None]
    mask[:, :border_width] = np.linspace(0.2, 1.0, border_width)
    mask[:, -border_width:] = np.linspace(1.0, 0.2, border_width)
    return mask

def search_boundary(image, center, initial_radius, directions):
    """Search for pupil boundaries in multiple directions from the center."""
    h, w = image.shape
    x_center, y_center = center
    boundary_points = []

    for dx, dy in directions:
        x, y = x_center, y_center
        for r in range(initial_radius, initial_radius + 50):  # Search outward
            x_new, y_new = int(x + r * dx), int(y + r * dy)
            if 0 <= x_new < w and 0 <= y_new < h:
                if image[y_new, x_new] > 50:  # Stop at a significant intensity
                    boundary_points.append((x_new, y_new))
                    break

    return boundary_points

def fit_ellipse_to_boundary(image, boundary_points):
    """Fit an ellipse to the detected boundary points."""
    if len(boundary_points) >= 5:  # OpenCV requires at least 5 points to fit an ellipse
        try:
            ellipse = cv2.fitEllipse(np.array(boundary_points))
            if ellipse[1][0] > 0 and ellipse[1][1] > 0:  # Ensure valid ellipse dimensions
                result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.ellipse(result_image, ellipse, (0, 0, 255), 2)
                return result_image, ellipse
            else:
                print("Invalid ellipse dimensions, skipping.")
                return image, None
        except cv2.error as e:
            print(f"Error fitting ellipse: {e}")
            return image, None
    else:
        print("Not enough boundary points to fit an ellipse.")
        return image, None

def refine_radius(image, center, initial_radius):
    """Refine the detected radius by shrinking it based on intensity criteria."""
    x, y = center
    current_radius = initial_radius
    
    while current_radius > 5:
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (x, y), current_radius, 255, -1)
        mean_intensity = cv2.mean(image, mask=mask)[0]
        if mean_intensity < 50:  # Adjust the threshold based on your dataset
            break
        current_radius -= 1
    
    return current_radius

def localize_pupil(image, candidate_mask, gradient_direction, border_mask, min_radius=10, max_radius=50):
    """Localize the pupil using the Hough transform on candidate points and elliptical fitting."""
    hough_space = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape
    
    try:
        for y in range(height):
            for x in range(width):
                if candidate_mask[y, x] > 0:
                    angle = gradient_direction[y, x]
                    for r in range(min_radius, max_radius + 1):
                        cx = int(x - r * np.cos(angle))
                        cy = int(y - r * np.sin(angle))
                        
                        if 0 <= cx < width and 0 <= cy < height:
                            hough_space[cy, cx] += border_mask[cy, cx]
        
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hough_space)
        best_center = maxLoc
        
        initial_radius = 0
        best_votes = 0
        for r in range(min_radius, max_radius + 1):
            votes = hough_space[maxLoc[1], maxLoc[0]]
            if votes > best_votes:
                initial_radius = r
                best_votes = votes
        
        # Refine the radius based on the assumption that the pupil is dark
        refined_radius = refine_radius(image, best_center, initial_radius)
        
        # Enhance the pupil detection by fitting an ellipse
        directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (-2, 0), (2, 0), (0, 2), (0, -2)]
        boundary_points = search_boundary(image, best_center, refined_radius, directions)
        
        # Debug: Draw boundary points
        for point in boundary_points:
            cv2.circle(image, point, 2, 255, -1)
        
        final_image, ellipse = fit_ellipse_to_boundary(image, boundary_points)
        
        return final_image, ellipse, hough_space
    except Exception as e:
        print(f"Error during pupil localization: {e}")
        return image, None, None