import pandas as pd
from src.embedding.local_embedding import LocalEmbedddings

def save_embeddings_to_file(csv_path, json_path, model_name):
    """
    Saves embeddings of product descriptions to a file.
    
    Args:
        csv_path (str): Path to the CSV file containing product descriptions.
        json_path (str): Path to save the JSON file with embeddings.
        model_name (str): Name of the embedding model to use.
    """
    df = pd.read_csv(csv_path)
    descriptions = df["description"].tolist()
    local_embeddings = LocalEmbedddings(model_name)
    embeddings = local_embeddings.encode_documents(descriptions, batch_size=2)
    df['embedding'] = [embedding.tolist() for embedding in embeddings]

    # 4. Select the desired columns
    output_df = df[['name', 'description', 'price', 'embedding', 'image_string']]

        # 5. Save the DataFrame to a JSON file
    output_df.to_json(json_path, orient='records', indent=4)
        
    print(f"Successfully saved embedded data to {json_path}")
    
if __name__ == "__main__":
    csv_path = "data/data_with_base64.csv"
    json_path = "data/embedded_data.json"
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    save_embeddings_to_file(csv_path, json_path, model_name)
    