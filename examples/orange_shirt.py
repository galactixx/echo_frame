from echoframe.embeddings import Vectorizer
from echoframe.media import MediaLoader
from echoframe.search import VectorSearchEngine
from echoframe.vector_store.vector_store import BasicVectorStore

def main(query: str) -> None:
    """
    Example showing the successful image search of the query ("man in an orange shirt") amongst ten images.
    Obviously a very simple example with few embeddings, but an illustration of the capability.
    """

    # Additional images to add into embeddings for example
    additional_images = [
        './examples/images/dog.jpg',
        './examples/images/woman_with_umbrella.jpg'
    ]

    # Instantiate media loader and converstion of media
    media_loader = MediaLoader(media_paths=[
        './examples/images/green_car.jpg',
        './examples/images/orange_shirt.jpg',
        './examples/images/orange_house.jpg',
        './examples/images/mona_lisa.jpg',
        './examples/images/homework.jpg',
        './examples/images/teddy_bear.jpg',
        './examples/images/joe_biden.jpg',
        './examples/images/code.png'
    ])

    # Initialize vectorizer model
    vectorizer = Vectorizer(
        frames=media_loader.media_frames
    )

    # Initialize vector store
    vector_store = BasicVectorStore(
        frames=vectorizer.frames,
        image_embeddings=vectorizer.initial_embeddings
    )

    # Initialize consolidated vector search engine
    vector_search = VectorSearchEngine(
        vectorizer=vectorizer,
        vector_store=vector_store
    )

    # Convert images/frames to embeddings and add to vector store
    vector_search.add(
        frames=MediaLoader(media_paths=additional_images).media_frames
    )

    # Query embeddings to get frame
    frame = vector_search.search(query=query)

    # Show queried image
    MediaLoader.show_image(image=frame[0][0])

if __name__ == "__main__":

    # Query vector database for "man in an orange shirt"
    main(query="man in an orange shirt")