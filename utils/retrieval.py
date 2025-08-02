import asyncio
import logging
from src.embedding.local_embedding import LocalEmbedddings
from chromadb import Collection
from typing import Dict, List
from chromadb import PersistentClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def retrieval_documents(
    query_texts: List[str], 
    collection: Collection, 
    embedding_model: LocalEmbedddings, 
    n_results: int = 5
) -> Dict:
    """
    Asynchronously retrieves documents from ChromaDB using a pre-initialized embedding model.
    
    Args:
        query_texts (List[str]): The query texts to search for.
        collection (Collection): The ChromaDB collection to search in.
        embedding_model (LocalEmbedddings): An initialized instance of the embedding model.
        n_results (int): The number of top results to return.
    
    Returns:
        Dict: The raw dictionary of results from ChromaDB.
    """
    try:
        loop = asyncio.get_running_loop()
        
        # Run the synchronous embedding encoding in a separate thread
        query_embeddings = await loop.run_in_executor(
            None, 
            lambda: embedding_model.encode_query(query_texts)
        )
        
        # Run the synchronous ChromaDB query in a separate thread
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(
                query_embeddings=[embedding.tolist() for embedding in query_embeddings],
                n_results=n_results
            )
        )
        return results
    except Exception as e:
        logger.error(f"An error occurred during document retrieval: {e}")
        raise

async def main_test():
    """Asynchronous main function for testing."""
    
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    client = PersistentClient(path="./chromadb")
    collection = client.get_or_create_collection(name="hypervibe_products")
    embedding_model = LocalEmbedddings(model_name)
    
    query = "a black skirt"
    results = await retrieval_documents([query], collection, embedding_model, n_results=3)
    
    # print(f"Query: '{query}'")
    # for doc in results['metadata'][0]:
    #     print(f"Document: {doc}")
        

if __name__ == "__main__":
    asyncio.run(main_test())