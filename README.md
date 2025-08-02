# Hypervibe-shopping

Hypervibe-shopping is a chat-based application that helps users discover and shop for items tailored to their styles and preferences. Users can interact with the service by sending text queries or attaching images. The system analyzes user input, along with their shopping history stored in our database, to suggest or recommend the most probable next items that match their tastes.

## Features
- **Conversational Interface:** Chat with the application to receive personalized shopping recommendations.
- **Image Recognition:** Attach images to get suggestions for similar or matching products.
- **Personalized Suggestions:** Recommendations are based on both user preferences and historical data.
- **Database Integration:** User history is securely stored and leveraged to improve recommendation accuracy.
- **Multilingual Support:** The application supports both Thai and English, allowing users to interact and receive recommendations in their preferred language.

## Endpoints
1. `/login`: Handles user authentication and session management.
2. `/detect`: Determines input language whether it is Thai or English.
3. `/translate`: Translates user input to English if it is initially provided in Thai.
4. `/caption`: Generates descriptive caption for an uploaded image.
5. `/history`: Retrieves the user's purchase and interaction history.
6. `/extract`: Extracts key information (user's desire, preferences, and history) from user input.
7. `/retrieve`: Retrieves recommended items in a hyper-personalized way.
8. `/summarize`: Summarizes recommendation results.
