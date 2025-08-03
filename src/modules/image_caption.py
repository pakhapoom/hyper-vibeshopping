import asyncio
import logging 
from PIL import Image 
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSOR = None
MODEL = None
try:
    logger.info("Loading BLIP-2 model and processor into memory...")
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    PROCESSOR = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
    MODEL = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", device_map="auto"
        #quantization_config=quantization_config
    )
    logger.info("BLIP-2 model and processor loaded successfully.")
except Exception as e:
    logger.error(f"Fatal error: Could not load the BLIP-2 model. {e}")

def _generate_caption_sync(raw_image: Image.Image) -> str:
    """Synchronous helper function that performs the actual model inference."""
    inputs = PROCESSOR(raw_image, return_tensors="pt").to(MODEL.device, torch.float16)
    out = MODEL.generate(**inputs, max_new_tokens=50)
    caption = PROCESSOR.decode(out[0], skip_special_tokens=True).strip()
    return caption

async def generate_caption(image: Image.Image) -> str:
    """
    Asynchronously generates a caption for a given PIL Image object.
    
    Args:
        image (Image.Image): A PIL Image object to be captioned.
    
    Returns:
        str: The generated caption for the image.
    """
    if not MODEL or not PROCESSOR:
        raise RuntimeError("Model is not loaded. Cannot generate caption.")

    try:
        rgb_image = image.convert('RGB')
        loop = asyncio.get_running_loop()
        caption = await loop.run_in_executor(
            None,  # Use the default thread pool executor
            _generate_caption_sync,
            rgb_image
        )
        return caption
    
    except Exception as e:
        logger.error(f"An error occurred during caption generation: {e}")
        raise RuntimeError(f"An error occurred during caption generation: {e}")

async def main_test():
    """Asynchronous main function for testing the module."""
    try:
        image_path = "data/Image33.png"
        with Image.open(image_path) as image:
            caption = await generate_caption(image)
            print(f"Generated Caption: {caption}")
    except FileNotFoundError:
        logger.error(f"Error: Test image not found at '{image_path}'")
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}")

if __name__ == "__main__":
    asyncio.run(main_test())