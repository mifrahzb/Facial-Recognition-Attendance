import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    vec1, vec2: numpy.ndarray
        Input vectors.

    Returns:
    float
        Cosine similarity value.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:  # Avoid division by zero
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

def match_embedding(new_embedding, db_embeddings, threshold=0.8):
    """
    Compare the new embedding against database embeddings using cosine similarity.

    Parameters:
    new_embedding: numpy.ndarray
        The real-time embedding to match.
    db_embeddings: list of numpy.ndarray
        List of embeddings from the database.
    threshold: float
        The threshold for cosine similarity to consider a match.

    Returns:
    list
        Indices of matching embeddings in the database.
    """
    matches = []
    for i, db_embedding in enumerate(db_embeddings):
        similarity = cosine_similarity(new_embedding, db_embedding)
        if similarity >= threshold:
            matches.append((i, similarity))  # Index and similarity score
    return matches
