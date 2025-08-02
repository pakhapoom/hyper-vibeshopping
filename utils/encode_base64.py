import os 
import base64
import pandas as pd

def encode_image_to_base64(image_path):
    """
    Encodes an image file to a base 64 string.
    
    Args:
        image_path (str): The path to the image file.
    Returns:
        str: The base 64 encoded string of the image.
    """
    try:
        image_path = os.path.join('/home/tam/hyper-vibeshopping/data', image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The file {image_path} does not exist.")
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_string = base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"An error occurred while encoding the image: {e}")
    return base64_string

if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    df["image_string"] = df["imageUrl"].apply(encode_image_to_base64)
    df = df.drop(columns=["imageUrl"])
    df.to_csv("data/data_with_base64.csv", index=False)
    

