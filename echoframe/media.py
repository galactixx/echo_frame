from typing import List
import os

import cv2
from cv2.typing import MatLike
from PIL import Image
import numpy as np

class MediaLoader:
    """
    Prepare media, either photos or videos, for conversion to embeddings.
    """
    def __init__(self, media_paths: List[str]):
        if not isinstance(media_paths, list):
            raise ValueError(
                'media_paths argument must be of type list'
            )

        self.media_frames = []

        for path in media_paths:
            path_lower = path.lower()
            path_extension = os.path.splitext(path_lower)[1]

            if path_extension == '.mp4':
                self.media_frames.append(
                    self._load_video(path=path)
                )

            elif path_extension in ['.jpg', '.png']:
                self.media_frames.append(
                    [self._load_image(path=path)]
                )
            
            else:
                raise Exception(f'file extension {path_extension} is not supported')
            
    @staticmethod
    def show_image(image: Image) -> None:
        """
        After resulting image, convert PIL image into cv2 MatLike and show.
        """

        # Convert PIL Image to NumPy array
        numpy_image = np.array(image)

        # Convert RGB to BGR
        matlike_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        # Now you can display it with OpenCV
        cv2.imshow('Image', matlike_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    def _load_image(self, path: str) -> Image:
        """
        Takes in a media path and loads an image as a MatLike object.
        """

        image = cv2.imread(path)
    
        # Convert BGR (OpenCV format) to RGB
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        frame_pil = Image.fromarray(frame_rgb)

        return frame_pil

    def _load_video(self, path: str) -> List[MatLike]:
        """
        Takes in a media path and loads frame by frame video as a MatLike object.
        """

        frames = []

        cap = cv2.VideoCapture(path)

        # Check if the video opened successfully
        if not cap.isOpened():
            raise Exception("Error: Could not open video.")

        while True:

            # Read the frame
            ret, frame = cap.read()
            frames.append(frame)

            if not ret:
                break

        cap.release()
        
        return frames