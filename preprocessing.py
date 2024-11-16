import cv2
import numpy as np

def process_image(face_region):
    """
    Process the given face region for further embedding generation.
    
    Parameters:
    face_region: The cropped face region. Can be a NumPy array or convertible to one.

    Returns:
    numpy.ndarray: The preprocessed image ready for embedding generation.
    """
    # Ensure the face_region is a NumPy array
    if not isinstance(face_region, np.ndarray):
        try:
            face_region = np.array(face_region)
        except Exception as e:
            raise ValueError("The input face_region cannot be converted to a NumPy array.") from e

    # Convert to grayscale if necessary
    if len(face_region.shape) == 3:  # Check if the image is in color
        gray_image = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = face_region

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(gray_image)

    # Apply sharpening filter
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    sharpened_image = cv2.addWeighted(equalized_image, 1.5, blurred_image, -0.5, 0)
    
    # Save or show the processed image (optional)
    cv2.imwrite("sharpened_image.jpg", sharpened_image)
    # cv2.imshow("Processed Image", sharpened_image)
    cv2.waitKey(1)  # Display for a short time
    cv2.destroyAllWindows()

    return sharpened_image
