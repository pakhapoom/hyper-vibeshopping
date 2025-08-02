from src.embedding.local_embedding import LocalEmbedddings

def retrieval_documents(model_name,query, collection, top_k):
    """
    Retrieves documents from the ChromaDB collection based on a query.
    
    Args:
        query (str): The query string to search for.
        collection (chromadb.Collection): The ChromaDB collection to search in.
        top_k (int): The number of top results to return.
    
    Returns:
        list: A list of retrieved documents and their metadata.
    """
    embedding_model = LocalEmbedddings(model_name)
    query_embedding = embedding_model.encode_query(query)
    
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )
    
    retrieved_docs = []
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        retrieved_docs.append({
            'document': doc,
            'metadata': metadata
        })
    
    return retrieved_docs

if __name__ == "__main__":
    from chromadb import PersistentClient
    import json
    
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    CHROMA_DB_PATH = "./chromadb"
    COLLECTION_NAME = "hypervibe_products"
    
    client = PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    query_text = "black sequin skirt"
    top_k = 3
    
    retrieved_docs = retrieval_documents(model_name,query_text, collection, top_k)
    
    print(f"Retrieved {len(retrieved_docs)} documents for query '{query_text}':")
    for doc in retrieved_docs:
        print(f"Name: {doc['metadata']['name']}, Price: {doc['metadata']['price']}")