from torch import Tensor
from abc import ABC, abstractmethod

from echoframe.typing import (
    Embeddings,
    MediaFrame
)

class BaseVectorStore(ABC):
    """
    Base interface for vector database.
    """
    @abstractmethod
    def add_vector(self, frames: MediaFrame, embeddings: Embeddings) -> None:
        pass

    @abstractmethod
    def subtract_vector(self) -> None:
        pass

    @abstractmethod
    def search_vectors(self, text_embedding: Tensor) -> MediaFrame:
        pass