import torch
from sentence_transformers import SentenceTransformer 
from tqdm import tqdm


class LocalEmbedddings:
    def __init__(self, model_name):
        """
        Initializes the LocalEmbedddings class with a specified model name.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
    
    def encode_documents(self, documents, batch_size=32):
        """
        Encodes documents (product descriptions) into embeddings.
        Args:
            documents (list): List of product descriptions.
        """
        if isinstance(documents, str):
            documents = [documents]
        
        all_embeddings = []
        
        # Process in batches with custom progress bar
        with tqdm(total=len(documents), desc="Encoding Documents", unit="docs") as pbar:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
                pbar.update(len(batch))
        
        return all_embeddings

    def encode_query(self, query):
        """
        Encodes a query into an embedding.
        Args:
            query (str): The query string to encode.
        """
        if isinstance(query, str):
            query = [query]
        embedding = self.model.encode(query)
        return embedding
    
if __name__ == "__main__":
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    local_embeddings = LocalEmbedddings(model_name)
    documents = ["This is a sample product description.", "Another product description."]
    embeddings = local_embeddings.encode_documents(documents, batch_size=2)
    print("Encoded Documents:", embeddings)
    