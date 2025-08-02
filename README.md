# team09: Hypervibe-shopping

Hypervibe-shopping is a chat-based application that helps users discover and shop for items tailored to their styles and preferences. Users can interact with the service by sending text queries or attaching images. The system analyzes user input, along with their shopping history stored in our database, to suggest or recommend the most probable next items that match their tastes.

## Features
- **Conversational Interface:** Chat with the application to receive personalized shopping recommendations.
- **Image Recognition:** Attach images to get suggestions for similar or matching products.
- **Personalized Suggestions:** Recommendations are based on both user preferences and historical data.
- **Database Integration:** User history is securely stored and leveraged to improve recommendation accuracy.
- **Multilingual Support:** The application supports both Thai and English, allowing users to interact and receive recommendations in their preferred language.

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
- `image_str`: `str`
