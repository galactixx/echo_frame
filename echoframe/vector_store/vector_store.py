import numpy as np
from torch import Tensor

from echoframe.vector_store.base import BaseVectorStore
from echoframe.typing import (
    Embeddings,
    MediaFrame
)

def intermediate_list_concat(base_array: np.ndarray, new_array: np.ndarray) -> np.ndarray:
    """
    Concat to an np.ndarray using an intermediate list.
    More efficient than using np.concatenate()
    """

    base_array_to_list = list(base_array)
    base_array_to_list.extend(new_array)
    base_array_new = np.array(base_array_to_list)

    return base_array_new

def normalize_embedding(embeddings: Tensor) -> Tensor:
    """
    Normalize a singular embedding vector.
    """

    norm = np.linalg.norm(embeddings)

    # Normalize the embedding if the norm is not zero
    if norm != 0:
        embeddings_norm = embeddings / norm
    else:
        embeddings_norm = embeddings
    
    return embeddings_norm

class BasicVectorStore(BaseVectorStore):
    """
    Basic vector store for video embeddings using cosine similarity for search.
    """
    def __init__(self, frames: MediaFrame = [], image_embeddings: Embeddings = []):
        self._frames = frames

        # Normalize embeddings
        self._embeddings = np.array([
            normalize_embedding(embedding) for embedding in image_embeddings
        ])

    def add_vector(self, frames: MediaFrame, embeddings: Embeddings) -> None:
        """
        Add new vector to image embeddings.
        """
        
        # Normalize embeddings provided
        embeddings_normalized = np.array([
            normalize_embedding(embedding) for embedding in embeddings
        ])

        # Add new embeddings to existing embeddings
        self._embeddings = intermediate_list_concat(
            base_array=self._embeddings, 
            new_array=embeddings_normalized
        )
        
        # Extend existing frames list
        self._frames.extend(frames)

    def subtract_vector(self) -> None:
        """
        Subtract existing vector from image embeddings.
        """
        raise NotImplementedError(
            "Subtraction of vectors is not yet implemented in this class."
        )

    def search_vectors(self, text_embedding: Tensor) -> MediaFrame:
        """
        Search vector store using cosine similarity.
        """

        # Normalize query embedding and calculate similarities
        text_embedding_norm = normalize_embedding(embeddings=text_embedding)
        scores = np.dot(text_embedding_norm, self._embeddings.T)

        # Generate the sorted indices based on scores
        sorted_top_indices = np.argsort(scores)[::-1][:1]

        # Retrieve the correct frames
        frames = [self._frames[i] for i in sorted_top_indices]

        return frames