import os
from dotenv import load_dotenv
from aift import setting
from aift.nlp import text_sum
from aift.multimodal import textqa
from aift.nlp.translation import th2en
from vllm import LLM, SamplingParams


load_dotenv()
aift_token = os.getenv("AIFT_TOKEN")
setting.set_api_key(aift_token)


prompt_template = {
    "detect": """Please analyze the following input text and determine whether it is written in Thai or English. Respond with only "Thai" or "English" based on your detection.
Input: {user_input}
Language: 
""",
    "translate": """Please translate the following Thai text into English. Respond with the translated text only, without any additional explanations or comments.
Input: {user_input}
Translation: 
""",
    "rewrite": """Please rewrite the following product description into a clear and concise search query that can be used to find similar clothing items and provide recommendations to users.
Note that you may use the user's preferences and previous purchases to enhance the search query.
Input: {user_input}
Product Description: {item_description}
Customer Data: {customer_data}
Search Query: 
""",
    "summarize": """Please summarize the following context which are the recommendations for clothing items based on the user's preferences. The summary should contain element showing personalization and sumarize the reccommendations in a concise manner.
Context: {context}
Summary: 
""",
}


def translate(user_input: str) -> str:
    res = th2en.translate(user_input)
    return res["translated_text"]

def generate(prompt: str) -> str:
    return textqa.generate(user_input, return_json=False)

def summarize(context: str) -> str:
    return text_sum.summarize(context)

class vLLMGenerator:
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-4B",
        temperature: float = 0.8,
        top_p: float = 0.95
    ):
        self.llm = LLM(model=model_name)
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p)

    def generate(self, prompt: str) -> str:
        output = self.llm.generate(prompt, self.sampling_params)
        return output.outputs[0].text


if __name__ == "__main__":
    user_input = "ช่วยแนะนำเสื้อผ้าเหมือนในรูปให้หน่อย"
    # user_input = "Can you recommend some clothes similar to those in the picture?"

    print("detect")
    print(generate(prompt_template["detect"].format(user_input=user_input)))

    print("\n\ntranslate")
    print("nlp:", translate(user_input))
    print("textqa:", generate(prompt_template["translate"].format(user_input=user_input)))

    print("\n\nrewrite")
    user_input = "Can you recommend some clothes similar to those in the picture?"
    item_description = "This vibrant sunny yellow linen shirt is designed for a relaxed and airy fit, perfect for warm weather. It features a classic collar and a single chest pocket, with the ability to be styled loosely or tucked in. The breathable linen fabric ensures comfort and a casually sophisticated look. Style Vibe: Resort Casual / Effortless Summer. Fit: Relaxed, slightly oversized. Material: Lightweight and breathable linen. Care: Machine wash cold, hang to dry. Matches Well With: Bottoms: Matching linen trousers (as implied), white denim jeans, or tailored chino shorts. Inner Tops: A simple white camisole or tank top if worn unbuttoned. Shoes: Flat sandals, espadrille wedges, or boat shoes. Accessories: A large straw beach bag or a wide-brimmed sun hat."
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
    print(generate(prompt_template["rewrite"].format(
        user_input=user_input,
        item_description=item_description,
        customer_data=customer_data,
    )))

    print("\n\nsummarize")
    context = """
- Women's White H-Strap Flat Sandals: These women's white flat sandals feature a distinctive 'H' strap design, offering a chic and minimalist look perfect for warm weather. Their open-toe style and comfortable sole make them ideal for casual outings and everyday wear. Style Vibe: Minimalist Chic / Summer Essential. Fit: True to size, comfortable flat sole. Material: Faux leather or synthetic material. Care: Wipe clean with a damp cloth. Matches Well With: Attire: Sundresses, shorts, skirts, or light trousers. Colors: Neutrals, pastels, or bright summer colors.
- Men's Navy Pinstripe Short-Sleeve Henley Shirt: This men's navy blue short-sleeve Henley shirt features classic white pinstripes and a band collar with a partial button placket. The relaxed fit and rolled-up sleeves offer a casual and comfortable aesthetic, making it suitable for warm weather and relaxed outings. Style Vibe: Casual Summer / Resort Wear. Fit: Relaxed fit, short-sleeve, band collar. Material: Lightweight cotton or linen blend. Care: Machine wash cold. Matches Well With: Bottoms: Light-colored chinos, linen shorts, or dark wash jeans. Shoes: Espadrilles, boat shoes, or casual sneakers. Accessories: Sunglasses or a woven bracelet.
- Men's Light Wash Baggy Denim Shorts: These men's light wash denim shorts feature a noticeably baggy fit that extends to just above the knee, offering a comfortable and relaxed silhouette. The distressed detailing and classic denim wash give them a casual, vintage-inspired streetwear appeal, perfect for warm weather. Style Vibe: 90s Streetwear / Skater Casual. Fit: Baggy fit, knee-length. Material: Cotton denim. Care: Machine wash cold. Matches Well With: Tops: An oversized graphic t-shirt (as shown) or a relaxed-fit hoodie. Shoes: High-top canvas sneakers (as shown) or skate shoes. Accessories: A baseball cap or a chain wallet.
- Men's Beige Short-Sleeve Linen Blend Shirt with Necklace: This men's short-sleeve shirt is a light beige or natural color, made from a textured linen blend fabric, and features a stand collar with a partial button placket. It is styled with a long, dark brown string necklace that has a decorative pendant. A single buttoned pocket is on the left chest. Style Vibe: Casual / Bohemian / Resort Wear. Fit: Regular fit short-sleeve shirt. Material: Linen blend. Care: Machine wash cold. Matches Well With: Bottoms: Light-colored shorts or chinos. Shoes: Sandals or espadrilles. Accessories: Woven bracelets.
- Elephant Festival Camp Shirt  Onyx Ivory: This breezy Thai rayon camp-collar shirt bursts with iconic elephant processions and lotus mandalas, designed to sync seamlessly with matching elephant pants for an effortlessly cohesive, culture-rich look. Style Vibe: Island Festival / Boho Street. Fit: Relaxed drape with a boxy cut and soft camp collarideal for layering or wearing open over swimwear. Material: Feather-light 100 % Thai rayoncool, quick-drying, and silky against sun-kissed skin. Care: Machine-wash cold on gentle; hang dry to keep prints crisp and fabric flowing. Matches Well With: Elephant Pantspair with coordinating Thai elephant harem or palazzo pants for a head-to-toe statement set; Footwearleather sandals, espadrilles, or barefoot on the sand; Accessorieswoven straw hat, vintage shades, layered beaded necklaces for festival-ready flair.
"""
    print(summarize(context))
