import os
from dotenv import load_dotenv
import asyncio
from aift import setting
from aift.nlp import text_sum
from aift.multimodal import textqa
from aift.nlp.translation import th2en
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


load_dotenv()
aift_token = os.getenv("AIFT_TOKEN")
setting.set_api_key(aift_token)


prompt_template = {
    "detect": """Please analyze the following input text and determine whether it is written in Thai or English. Respond with only "Thai" or "English" based on your detection.

# Examples 1
Input: ช่วยแนะนำเสื้อผ้าเหมือนในรูปให้หน่อย
Language: Thai

# Examples 2
Input: Can you recommend some clothes similar to those in the picture?
Language: English

Input: {user_input}
Language: 
""",

    "translate": """Please translate the following Thai text into English. Respond with the translated text only, without any additional explanations or comments.
Input: {user_input}
Translation: 
""",

    "rewrite": """Please extract user preferences from customer data and product description, and rewrite the user input into a clear query containing user desire, preferences, and product details.
Respond with the rewritten query only, without any additional explanations or comments.

# Example
Input: Please recommend clothes like the ones in the picture
Product Description: "A white T-shirt with the text 'AI for Thai Hackathon 2025'"
Customer Data: # Customer Information
- Gender: M
- Age: 32
- Occupation: businessman
- Address: rayong

# Purchase History (The most recent 5 purchases)
- Hack Day Matrix Tee  Black & White
- Hack Life Tri-Blend Tee  Violet Burst
- Impact Hackathon Tech Tee  Circuit Black
- Intel Hackathon 2022 Hoodie Coder Tee  Midnight Black
- Tidewhite Relaxed Linen Set  Classic White:
Rewritten Query: Please recommend clothes similar to the white T-shirt with the text 'AI for Thai Hackathon 2025', considering the customer's preferences for casual and comfortable clothing suitable for a businessman in Rayong, who has previously purchased items like linen suits and shirts.

Input: {user_input}
Product Description: {item_description}
Customer Data: {customer_data}
Rewritten Query: 
""",

    "summarize": """Please summarize the following context which are the recommendations for clothing items based on the user input. The summary should contain element showing personalization and summarize the recommendations in a concise manner.

# Example
Input: Please recommend a t-shirt that suitable for an AI scientist with age around 25.
Context: Hack Day Matrix Tee Black & White: Tech-inspired tee with bold matrix print for hackathons. Style: Geek Chic. Fit: Relaxed unisex crew. Material: 100 percent ring-spun cotton. Care: Cold wash, low tumble. Pairs well with jeans, hoodie, and laptop bag.
Hack Life Tri-Blend Tee Violet Burst: Vibrant violet tee with fun coding icons. Style: Startup Casual. Fit: Soft unisex drape. Material: Tri-blend (poly/cotton/rayon). Care: Cold wash, low tumble. Pairs well with joggers, hoodie, and stickered laptop.
Impact Hackathon Tech Tee Circuit Black: Futuristic tee with neon circuit design from 2020 Impact Hackathon. Style: Techwear. Fit: Athletic unisex. Material: Moisture-wicking polyester blend. Care: Cold wash, low tumble. Pairs well with joggers, sneakers, and smartwatch.
Intel Hackathon 2022 Hoodie Coder Tee Midnight Black: Features a hooded coder silhouette with binary glow. Style: Dark Mode. Fit: Regular unisex crew. Material: 100 percent soft-spun cotton. Care: Cold wash inside out, low tumble. Pairs well with black jeans and headphones.
Tidewhite Relaxed Linen Set Classic White: Breezy linen set for beach weddings. Style: Boho Coastal. Fit: Relaxed and airy. Material: Lightweight linen-blend. Care: Cold wash, hang dry. Pairs well with sandals, straw hat, and optional blazer.
Summary: Based on your preferences, the recommendations offer a mix of tech-savvy style and functional comfort tailored to a youthful, innovative lifestyle:
Hack Day Matrix Tee: A black-and-white, matrix-themed shirt ideal for hackathons. It blends geek chic with a relaxed fit, perfect for pairing with jeans and a hoodie—great for casual coding sessions.
Hack Life Tri-Blend Tee: A soft, violet tee with playful coding icons. Its startup casual vibe suits a young tech professional, especially when worn with joggers and a stickered laptop.
Impact Hackathon Tech Tee: With its neon circuit design and moisture-wicking fabric, this athletic-fit shirt appeals to someone active and futuristic in their fashion—ideal for long hack days or tech meetups.
Intel Hackathon Hoodie Tee: Featuring a coder silhouette and binary glow, this dark-mode inspired tee offers a more understated, moody aesthetic—great for night owls or low-key dev environments.
Tidewhite Linen Set: While less relevant to the tech field, this breezy white linen set adds a stylish option for off-duty events like beach weddings or retreats.

Personalized takeaway: These picks balance tech culture with comfort and personality, matching the lifestyle of a young, creative AI scientist who values both function and flair.
    
Input: {rewrite}
Context: {context}
Summary: 
""",
}


async def translate(user_input: str) -> str:
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        None,  # Use the default thread pool executor
        th2en.translate,
        user_input
    )
    return res["translated_text"]

async def generate(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        None,
        textqa.generate,
        prompt
    )
    return res["content"]

async def summarize(context: str) -> str:
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        None,
        text_sum.summarize,
        context
    )
    return res["content"]

# class vLLMGenerator:
#     def __init__(
#         self, 
#         model_name: str = "Qwen/Qwen3-0.6B",
#         temperature: float = 0.8,
#         top_p: float = 0.95,
#         gpu_memory_utilization: float = 0.85,
#         max_model_len: int = 39840,
#     ):
#         self.llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
#         self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p)

#     def generate(self, prompt: str) -> str:
#         outputs = self.llm.generate(prompt, self.sampling_params)
#         generated_text = outputs[0].outputs[0].text
#         return generated_text

class TransformersGenerator:
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-1.7B",
    ):
        self.device = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def _generate_sync(self, prompt: str) -> str:
        # This is the synchronous blocking part
        messages = [{"role": "user", "content": prompt}]
        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([full_prompt], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content

    async def generate(self, prompt: str) -> str:
        # Asynchronous wrapper
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, prompt
        )


if __name__ == "__main__":

    test_module = "detect"  # Options: detect, translate, rewrite, summarize, rewrite_tfm, summarize_tfm

    # inputs
    user_input = "ช่วยแนะนำเสื้อผ้าเหมือนในรูปให้หน่อย"
    # user_input = "Can you recommend some clothes similar to those in the picture?"

    customer_data = """
# Customer Information
- Gender: M
- Age: 32
- Occupation: businessman
- Address: rayong

# Purchase History (The most recent 5 purchases)
- Shoreline Linen Wedding Suit  Sand Beige
- Men's White Stand-Collar Linen Shirt
- Men's Brown Loafer Shoes with Buckle Detail
- Men's Navy Blue Pleated Trousers
- Regatta Crest Beach Blazer Set  Navy & White
"""

    item_description = "This vibrant sunny yellow linen shirt is designed for a relaxed and airy fit, perfect for warm weather. It features a classic collar and a single chest pocket, with the ability to be styled loosely or tucked in. The breathable linen fabric ensures comfort and a casually sophisticated look. Style Vibe: Resort Casual / Effortless Summer. Fit: Relaxed, slightly oversized. Material: Lightweight and breathable linen. Care: Machine wash cold, hang to dry. Matches Well With: Bottoms: Matching linen trousers (as implied), white denim jeans, or tailored chino shorts. Inner Tops: A simple white camisole or tank top if worn unbuttoned. Shoes: Flat sandals, espadrille wedges, or boat shoes. Accessories: A large straw beach bag or a wide-brimmed sun hat."

    context = """
- Men's Grey Long-Sleeve Button-Down Shirt: This classic men's long-sleeve button-down shirt comes in a versatile grey shade, offering a clean and polished look. It features a traditional collar, full button placket, and a single chest pocket.
- Men's Navy Pinstripe Short-Sleeve Henley Shirt: This men's navy blue short-sleeve Henley shirt features classic white pinstripes and a band collar with a partial button placket. The relaxed fit and rolled-up sleeves offer a casual and comfortable aesthetic, making it suitable for warm weather and relaxed outings. Style Vibe: Casual Summer / Resort Wear. Fit: Relaxed fit, short-sleeve, band collar. Material: Lightweight cotton or linen blend. Care: Machine wash cold. Matches Well With: Bottoms: Light-colored chinos, linen shorts, or dark wash jeans. Shoes: Espadrilles, boat shoes, or casual sneakers. Accessories: Sunglasses or a woven bracelet.
- Men's Beige Short-Sleeve Linen Blend Shirt with Necklace: This men's short-sleeve shirt is a light beige or natural color, made from a textured linen blend fabric, and features a stand collar with a partial button placket. It is styled with a long, dark brown string necklace that has a decorative pendant. A single buttoned pocket is on the left chest. Style Vibe: Casual / Bohemian / Resort Wear. Fit: Regular fit short-sleeve shirt. Material: Linen blend. Care: Machine wash cold. Matches Well With: Bottoms: Light-colored shorts or chinos. Shoes: Sandals or espadrilles. Accessories: Woven bracelets.
- Men's Blue Cat Graphic Crewneck Sweatshirt: This light blue men's crewneck sweatshirt features an adorable graphic print of a cat wearing a backpack, with Japanese text on the side.
- Men's Black and Grey Plaid Long-Sleeve Shirt: This men's long-sleeve shirt features a classic black and grey plaid pattern, offering a timeless and versatile look. It has a traditional collar and full button-down front, perfect for layering over a t-shirt or wearing on its own for a casual yet put-together style.
"""
    
    rewrite = "Please recommend a linen shirt in sunny yellow color, relaxed and airy fit, suitable for a businessman in Rayong, matching linen trousers, white denim jeans, or tailored chino shorts, with a classic collar, single chest pocket, breathable linen fabric, machine wash cold, hang to dry, and style as resort casual or effortless summer, slightly oversized fit, and lightweight and breathable material, considering the customer's preferences for casual and comfortable clothing."


    if test_module == "detect":
        print("detect")
        print(generate(prompt_template["detect"].format(user_input=user_input)))

    elif test_module == "translate":
        print("translate")
        print("nlp:", translate(user_input))
        print("textqa:", generate(prompt_template["translate"].format(user_input=user_input)))

    elif test_module == "rewrite":
        print("rewrite")
        user_input = "Can you recommend some clothes similar to those in the picture?"
        print(generate(prompt_template["rewrite"].format(
            user_input=user_input,
            item_description=item_description,
            customer_data=customer_data,
        )))

    elif test_module == "summarize":
        print("summarize")
        print(summarize(context))

    elif test_module == "rewrite_tfm":
        print("TransformersGenerator rewrite")
        transformers_generator = TransformersGenerator()
        print(transformers_generator.generate(prompt_template["rewrite"].format(
            user_input=user_input,
            item_description=item_description,
            customer_data=customer_data,
        )))

    elif test_module== "summarize_tfm":
        print("TransformersGenerator summarize")
        transformers_generator = TransformersGenerator()
        print(transformers_generator.generate(prompt_template["summarize"].format(
            rewrite=rewrite,
            context=context,
        )))
