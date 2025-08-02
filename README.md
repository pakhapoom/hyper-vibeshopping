# team09: Hypervibe-shopping
Hypervibe-shopping is a chat-based application that helps users discover and shop for items tailored to their styles and preferences. Users can interact with the service by sending text queries or attaching images. The system analyzes user input, along with their shopping history stored in our database, to suggest or recommend the most probable next items that match their tastes.


## Features
- **Conversational Interface:** Chat with the application to receive personalized shopping recommendations.
- **Multilingual Support:** The application supports both Thai and English, allowing users to interact and receive recommendations in their preferred language.
- **Image Captioning:** Allow users to attach images to get suggestions for similar or matching products.
- **Preferences Integration:** User demographics as well as purchase history securely stored and leveraged to improve recommendation accuracy.
- **Personalized Suggestions:** Recommendations are based on both user preferences and historical data.


## Endpoints
### 1. `/login`
Handles user authentication.

**Input:** 
- `email`: `str`
- `password`: `str`

**Output:**
- `authentication_result`: `Dict[str, bool | str]`

### 2. `/chat`
Suggests recommended items in a hyper-personalized way.

**Input:** 
- `user_input`: `str`

**Output:**
- `ai_recommendation`: `str`

### 3. `/upload`
Handles image upload for product discovery and recommendations.

**Input:**  
- `image_file`: `file` (supported formats: JPEG, PNG)

**Output:**  
- `base64_str`: `str`

### 4. `/health`
Checks endpoint to confirm the service is running.

**Output:**  
- `health`: `Dict[str, str]`


## Postman collection
Please refer to [this json file](https://gitlab.nectec.or.th/hackathon/ai-thailand-2025/team09/-/blob/main/postman_collection.json?ref_type=heads) for the postman collection.


## AI4Thai services
1. Text generation: use `textqa.generate` to identify language of the user input.
2. Translation: use `th2en.translate` to translate Thai user input to English.


## Challenges in this project
1. Handling image input may need to convert to base64 string before processing further.
2. Deployment with docker is quite challenging for us.
