import chromadb
import pandas as pd
import json
from tqdm import tqdm

def batch_add_to_chroma(collection, documents, embeddings, metadatas, ids, batch_size):
    """
    Adds documents to the ChromaDB collection in batches.
    
    Args:
        collection: The ChromaDB collection to add documents to.
        documents (list): List of documents to add.
        embeddings (list): List of embeddings corresponding to the documents.
        metadatas (list): List of metadata for each document.
        ids (list): List of IDs for each document.
        batch_size (int): Size of each batch for adding documents.
    """
    for i in tqdm(range(0, len(documents), batch_size), desc="Adding to ChromaDB", unit="batch"):
        batch_docs = documents[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
    
def set_up_vector_db(chroma_db_path, collection_name, json_file_path):
    """
    Sets up the vector database by reading data from a JSON file and adding it to a ChromaDB collection.
    
    Args:
        chroma_db_path (str): Path to the ChromaDB directory.
        collection_name (str): Name of the collection to create or use.
        json_file_path (str): Path to the JSON file containing embedded data.
    """
    client = chromadb.PersistentClient(path=chroma_db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Using ChromaDB at {chroma_db_path} with collection '{collection_name}'")
    
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        
    embeddings = []
    documents = []
    metadatas = []
    ids = []
    
    for i, item in enumerate(data):
        documents.append(item['description'])
        embeddings.append(item['embedding'])
        metadatas.append({
            'name': item['name'],
            'price': item['price'],
            'image_url': item['imageUrl']  # Assuming 'imageUrl' is the correct key
        })
        ids.append(f"product_{i}")
    
    batch_size = 100
    batch_add_to_chroma(collection, documents, embeddings, metadatas, ids, batch_size)
    
    print(f"Successfully added {len(documents)} documents to ChromaDB collection '{collection_name}'")





if __name__ == "__main__":
    JSON_FILE_PATH = "data/embedded_data.json"
    CHROMA_DB_PATH = "./chromadb"
    COLLECTION_NAME = "hypervibe_products"
    set_up_vector_db(CHROMA_DB_PATH, COLLECTION_NAME, JSON_FILE_PATH)
    # print(f"Successfully added {len(documents)} documents to ChromaDB collection '{COLLECTION_NAME}'")
    # print("\nData upload complete!")
    # print(f"Total items in collection: {collection.count()}")
    # print("\nTesting with a sample query...")
    # query_text = "black sequin skirt"
    # embedding_model = LocalEmbedddings("Qwen/Qwen3-Embedding-0.6B")
    # query_embedding = embedding_model.encode_query(query_text)
    
    # # Use query_embeddings instead of query_texts
    # results = collection.query(
    #     query_embeddings=query_embedding.tolist(),  # Convert numpy array to list if needed
    #     n_results=2
    # )
    
    # print(f"Found {len(results['documents'][0])} results for '{query_text}'")
    # if results['documents'][0]:
    #     print(f"Top result: {results['metadatas'][0][0]['name']}")
    #     print(f"Price: {results['metadatas'][0][0]['price']}")
    
           
                                                 