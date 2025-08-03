import asyncio
import logging 
from typing import Dict, Optional, Text
from PIL import Image
import io
import re
import base64
from pandas import DataFrame
from src.modules.image_caption import generate_caption
from utils.retrieval import retrieval_documents
from src.embedding.local_embedding import LocalEmbedddings
from chromadb import PersistentClient
from src.modules.llm import translate, generate, summarize,prompt_template, TransformersGenerator
from src.modules.history import get_purchase_history


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
img_path = "uploads/temp_img.png"


class multimodal_search_service:
    """
    A service for multimodal search that combines image and text retrieval.
    Initializes all necessary models and clients once.
    """
    def __init__(
        self, 
        chroma_db_path: str = "./chromadb", 
        collection_name: str = "hypervibe_products",
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        use_transformer: bool = True,
    ):
        logger.info("Initializing multimodal search service...")
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.user_transformer = use_transformer 
        # Initialize ChromaDB client and collection
        self.client = PersistentClient(path=self.chroma_db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        # Initialize the text embedding model once
        self.embedding_model = LocalEmbedddings(embedding_model_name)
        logger.info("Service initialized successfully.")

        if self.user_transformer:
            logger.info("Using transformer for text generation.")
            self.transformer_generator = TransformersGenerator()


    @staticmethod
    def _check(text: str) -> str:
        if re.search(r'[a-zA-Z]', text):
            return "English"
        else:
            return "Thai"


    def check_language(self, user_input: str) -> str:
        """
        Detect the language of the input text.

        Args:
            user_input (str): The input text to analyze.

        Returns:
            str: "Thai" or "English" based on the detected language.
        """
        try:
            return generate(prompt_template["detect"].format(user_input=user_input))
        except:
            return self._check(user_input)
    

    def process_text(self, user_input: str) -> str:
        """
        Main method to handle text processing before RAG.

        Args:
            user_input (str): The input text to process.

        Returns:
            str: Clean text to send to LLM.
        """
        language = self.check_language(user_input)
        logger.info(f"Detected language: {language}")

        if language == "Thai":
            logger.info("Translating Thai text to English...")
            translated_text = translate(user_input)
            logger.info(f"Translated text: {translated_text}")
            return translated_text
        else:
            return user_input
    

    def rewrite_query(
        self,
        user_input: str,
        caption: str,
        cust_info: DataFrame,
    ):
        customer_data = get_purchase_history(cust_info)
        if self.user_transformer:
            rewrite = self.transformer_generator.generate(
                prompt_template["rewrite"].format(
                user_input=user_input,
                item_description=caption,
                customer_data=customer_data,
            ))
        else:
            rewrite = generate(
                prompt_template["rewrite"].format(
                user_input=user_input,
                item_description=caption,
                customer_data=customer_data,
            ))
        return rewrite
    

    def generate_answer(self, rewrite:str, context: str) -> str:
        if self.user_transformer:
            summary = self.transformer_generator.generate(
                prompt_template["summarize"].format(rewrite=rewrite,context=context)
            )
        else:
            summary = summarize(context)
        return summary


    async def search_by_image(
        self,
        user_input: str,
        cust_info: DataFrame,
        top_k: int = 5,
    ) -> Optional[Dict]:
        """
        Performs a search based on an input image by generating a caption
        and using it to query the vector database.
        """
        logger.info("Step 0: Preparing user input.")
        user_input = self.process_text(user_input)

        try:
            logger.info("Step 1: Getting the uploaded image.")
            # image = Image.open(img_path)
            with open(img_path, "rb") as f:
                image = f.read()

            logger.info("Step 2: Generating caption for the image...")
            caption = await generate_caption(image)
            logger.info(f"- Generated caption: '{caption}'")

            if not caption:
                logger.warning("Caption generation failed. Aborting search.")
                return None
            
            logger.info("Step 3: Rewriting query based on caption and user input.")
            cust_info = DataFrame(cust_info) if not isinstance(cust_info, DataFrame) else cust_info
            rewrite = self.rewrite_query(
                user_input=user_input,
                caption=caption,
                cust_info=cust_info,
            )

            logger.info(f"Step 4: Retrieving top {top_k} documents based on caption.")
            retrieved_results = await retrieval_documents(
                query_texts=[rewrite],
                collection=self.collection,
                embedding_model=self.embedding_model,
                n_results=top_k 
            )

            logger.info(f"Step 5: Summarizing the recommendation results.")
            documents = retrieved_results.get('documents', [[]])[0]
            metadatas = retrieved_results.get('metadatas', [[]])[0]

            context = "Here are recommendation results\n"
            for i, doc in enumerate(documents):
                context += f"Result {i+1}:\n"
                context += f"  - Item name: {metadatas[i]['name']}\n\n"
                context += f"  - Item description: {doc}\n"
            
            logger.info("Step 6: Generating final answer based on rewritten query and context.")
            summary = self.generate_answer(rewrite,context)

            return retrieved_results, summary

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
    cust_info = DataFrame([{'customer_id': 2, 'first_name': 'Oui', 'last_name': 'Pakhapoom', 'email': 'oui@aift.in.th', 'password': 1234, 'age': 32, 'occupation': 'businessman', 'address': 'rayong', 'gender': 'M'}])
    
    # 2. Prepare a test image by loading and encoding it to base64
    test_image_path = "data/Image33.png" # Make sure this path is correct
    try:
        with open(test_image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        
        logger.info(f"Successfully loaded and encoded test image: {test_image_path}")
        
        # 3. Call the search method
        results, summary = await service.search_by_image(image_b64, cust_info, top_k=3)
        
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

        print("-" * 40)
        print(f"Summary: {summary}")

    except FileNotFoundError:
        logger.error(f"Test image not found at '{test_image_path}'. Please check the path.")
    except Exception as e:
        logger.error(f"An error occurred during the service test: {e}", exc_info=True)


if __name__ == "__main__":
    # This allows you to run the test by executing the script directly
    asyncio.run(main_test())
