import asyncio
import logging 
from typing import Dict, Optional, Text
from PIL import Image
import io
import base64

from src.modules.image_caption import generate_caption
from utils.retrieval import retrieval_documents
from src.embedding.local_embedding import LocalEmbedddings
from chromadb import PersistentClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class multimodal_search_service:
    """
    A service for multimodal search that combines image and text retrieval.
    Initializes all necessary models and clients once.
    """
    def __init__(self, 
                 chroma_db_path: str = "./chromadb", 
                 collection_name: str = "hypervibe_products",
                 embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        
        logger.info("Initializing multimodal search service...")
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client and collection
        self.client = PersistentClient(path=self.chroma_db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        # Initialize the text embedding model once
        self.embedding_model = LocalEmbedddings(embedding_model_name)
        logger.info("Service initialized successfully.")
        
    async def search_by_image(self, image_b64: Text, top_k: int = 5) -> Optional[Dict]:
        """
        Performs a search based on an input image by generating a caption
        and using it to query the vector database.
        """
        try:
            logger.info("Step 1: Decoding base64 image.")
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))

            logger.info("Step 2: Generating caption for the image...")
            caption = await generate_caption(image)
            logger.info(f"  - Generated caption: '{caption}'")

            if not caption:
                logger.warning("Caption generation failed. Aborting search.")
                return None

            logger.info(f"Step 3: Retrieving top {top_k} documents based on caption.")
            retrieved_results = await retrieval_documents(
                query_texts=[caption],
                collection=self.collection,
                embedding_model=self.embedding_model,
                n_results=top_k 
            )
            
            return retrieved_results

        except base64.binascii.Error as e:
            logger.error(f"Base64 decoding error: {e}. Input may be malformed.")
            return None
        except Exception as e:
            logger.error(f"An error occurred during the search_by_image process: {e}", exc_info=True)
            return None
async def main_test():
    """Asynchronous main function for testing the service."""
    logger.info("--- Running Service Test ---")
    
    # 1. Initialize the service
    # This will also initialize the embedding model and ChromaDB client.
    service = multimodal_search_service()
    
    # 2. Prepare a test image by loading and encoding it to base64
    test_image_path = "data/Image33.png" # Make sure this path is correct
    try:
        with open(test_image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        
        logger.info(f"Successfully loaded and encoded test image: {test_image_path}")
        
        # 3. Call the search method
        results = await service.search_by_image(image_b64, top_k=3)
        
        # 4. Print the results in a clean format
        if results:
            print("\n--- Search Results ---")
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]

            if not documents:
                print("No documents found.")
            else:
                for i, doc in enumerate(documents):
                    print(f"Result {i+1}:")
                    print(f"  - Distance: {distances[i]:.4f}")
                    print(f"  - Name: {metadatas[i]['name']}")
                    print(f" - Price: {metadatas[i]['price']}")
            print("----------------------\n")
        else:
            print("Search returned no results or an error occurred.")

    except FileNotFoundError:
        logger.error(f"Test image not found at '{test_image_path}'. Please check the path.")
    except Exception as e:
        logger.error(f"An error occurred during the service test: {e}", exc_info=True)

if __name__ == "__main__":
    # This allows you to run the test by executing the script directly
    asyncio.run(main_test())