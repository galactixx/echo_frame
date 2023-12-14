from echoframe.typing import MediaFrame
from echoframe.vector_store.base import BaseVectorStore
from echoframe.embeddings import Vectorizer

class VectorSearchEngine:
    """
    Engine that combines the two separate embeddings and vector database functionality.
    Consolidates generating embeddings in combination with searching vector database. 
    """
    def __init__(
        self,
        vectorizer: Vectorizer,
        vector_store: BaseVectorStore
    ):
        self._vectorizer = vectorizer
        self._vector_store = vector_store
    
    def add(self, frames: MediaFrame) -> None:
        """
        Add more media data into vector store.
        """

        # Generate additional embeddings with new media data
        new_frames, new_embeddings = self._vectorizer.add_embeddings(frames=frames)  

        # Add embeddings to vector store
        self._vector_store.add_vector(
            frames=new_frames,
            embeddings=new_embeddings
        )

    def search(self, query: str) -> MediaFrame:
        """
        Utilizing the chosen vector store and vectorizer, searches vector store for most similar observations.
        """

        # Tokenize query using vectorizer
        query_embedding = self._vectorizer.query_embeddings(query=query)

        # Search existing vector store for most similar
        most_similar = self._vector_store.search_vectors(
            text_embedding=query_embedding
        )

        return most_similar