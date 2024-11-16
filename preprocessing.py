import cv2
import numpy as np

def process_image(face_region):
    """
    Process the given face region for further embedding generation.
    
    Parameters:
    face_region (numpy.ndarray): The cropped face region.

    Returns:
    numpy.ndarray: The preprocessed image ready for embedding generation.
    """
    # Convert to grayscale if necessary
    if len(face_region.shape) == 3:
        gray_image = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = face_region

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(gray_image)

    # Apply sharpening filter
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    sharpened_image = cv2.addWeighted(equalized_image, 1.5, blurred_image, -0.5, 0)
    cv2.imwrite("sharpened_image.jpg", sharpened_image)
    # cv2.imshow("Processed Image", sharpened_image)
    cv2.waitKey(1)  # Display for a short time
    cv2.destroyAllWindows()
    return sharpened_image
    