from typing import Tuple
import time
from urllib.error import URLError

from torch import Tensor
import torch
import clip

from echoframe.typing import (
    Embeddings, 
    MediaFrame
) 

class Vectorizer:
    """
    Generate and store CLIP embeddings to be used for querying.
    """
    def __init__(self, frames: list = []):
        self.frames = frames
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            self._model, self._preprocess = clip.load(
                "ViT-B/32", device=self._device
            )

        except URLError as e:
            print(f"An error occurred: {e}")

            time.sleep(5)
            self._model, self._preprocess = clip.load(
                "ViT-B/32", device=self._device
            )

        # Vectorize initial embeddings
        _, self.initial_embeddings = self.add_embeddings(
            frames=self.frames
        )

    def add_embeddings(self, frames: MediaFrame) -> Tuple[MediaFrame, Embeddings]:
        """
        Returns frames and embeddings of media provied (image or video).
        Converted to image embeddings using CLIP embeddings model.
        """

        embeddings = []
        
        with torch.no_grad():

            # Iterate through sets of frames
            for frame_set in frames:

                # Iterate through frames within set
                for frame in frame_set:
                    image = self._preprocess(frame).unsqueeze(0).to(self._device)
                    image_embedding = self._model.encode_image(image)

                    embeddings.extend(image_embedding)

        return frames, embeddings

    def query_embeddings(self, query: str) -> Tensor:
        """
        Tokenize query and conver to text embedding using CLIP emsbeddings model.
        """

        # Tokenize query
        text = clip.tokenize([query]).to(self._device)

        with torch.no_grad():
            text_embedding = self._model.encode_text(text).squeeze()

        return text_embedding