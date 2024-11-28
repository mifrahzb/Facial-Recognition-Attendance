import torch
from PIL import Image
import numpy as np
from torchvision import transforms, models
import torch.nn as nn

# Load pretrained ResNet-50 model from torchvision
resnet_model = models.resnet50(pretrained=True)
# Remove the final fully connected layer to use the embeddings
resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])  # Remove the classifier (fc) layer
resnet_model.eval()  # Set to evaluation mode to disable dropout, batchnorm updates, etc.

# Define transformation to resize, convert to tensor, and normalize the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50 standard input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet-50 normalization
])

# Define a function to get embeddings from ResNet-50 for a given face image
def get_face_embedding(face_image):
    """
    Convert a preprocessed face image into an embedding using ResNet-50 model.

    Parameters:
    face_image (numpy.ndarray or PIL.Image): The preprocessed face image in RGB format.

    Returns:
    numpy.ndarray: The embedding vector representing the face.
    """
    # Ensure image is in PIL format (required by PyTorch transforms)
    if isinstance(face_image, np.ndarray):
        face_image = Image.fromarray(face_image)

    # Ensure the image is in RGB format
    face_image = face_image.convert('RGB')
    
    # Apply transformations and add batch dimension
    face_tensor = transform(face_image).unsqueeze(0)  # Add batch dimension

    # Get embeddings without computing gradients
    with torch.no_grad():
        embedding = resnet_model(face_tensor)

    # Convert the embedding tensor to a numpy array and remove the batch dimension
    embedding_np = embedding.squeeze().numpy()
    return embedding_np
