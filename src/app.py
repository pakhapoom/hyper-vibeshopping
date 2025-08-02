import os
import base64
import logging
from uuid import uuid4
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Body, File, UploadFile
from pydantic import BaseModel, Field
from src.service import multimodal_search_service
from src.modules.login import login


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ImageSearchRequest(BaseModel):
    """Defines the structure of the incoming API request."""
    user_input: str = Field(..., description="User's input text for search context.")
    cust_info: List[Dict[str, Any]] = Field(..., description="Customer information for personalized search.")
    image_b64: str = Field(..., description="Base64 encoded string of the image file.")
    top_k: int = Field(5, gt=0, le=20, description="The number of top results to return.")

class SearchResultItem(BaseModel):
    """Defines the structure for a single item in the search results."""
    id: str
    distance: float
    metadata: Dict[str, Any]
    document: Optional[str] = None

class ImageSearchResponse(BaseModel):
    """Defines the structure of the API response."""
    results: List[SearchResultItem]
    generated_caption: str
    summary: str

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    authentication: bool
    cust_info: List[Dict[str, Any]]
    
class UploadRequest(BaseModel):
    image: UploadFile

class UploadResponse(BaseModel):
    base64_str: str


# --- FastAPI App Initialization ---

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app instance
app = FastAPI(
    title="HyperVibe Shopping Search API",
    description="API for multimodal product search using image-to-text and vector retrieval.",
    version="1.0.0"
)

# --- Global Service Instance ---
# The service is initialized once when the application starts.
# This ensures that all models and database clients are loaded into memory only one time.
try:
    service = multimodal_search_service()
    logger.info("Multimodal search service initialized successfully and is ready to accept requests.")
except Exception as e:
    logger.error(f"Fatal error during service initialization: {e}", exc_info=True)
    service = None


# --- API Endpoint ---
@app.post("/login")
def login_endpoint(request: LoginRequest):
    result = login(email=request.email, password=request.password)

    if not result["authentication"]:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return LoginResponse(
        authentication=result["authentication"], 
        cust_info=result["cust_info"]
    )


@app.post("/chat", response_model=ImageSearchResponse)
async def search_by_image_endpoint(request: ImageSearchRequest = Body(...)):
    """
    Accepts a base64 encoded image, generates a caption, and performs a vector search
    to find the most similar products.
    """
    if not service:
        raise HTTPException(status_code=503, detail="Service is unavailable due to an initialization error.")

    try:
        # Call the service's search method
        search_results, summary = await service.search_by_image(image_b64=request.image_b64, top_k=request.top_k, user_input= request.user_input, cust_info=request.cust_info)
        print(f"Search results: {search_results}")
        if search_results is None:
            raise HTTPException(status_code=404, detail="Could not find any results. The image might not have generated a valid caption.")

        # The caption is not directly available in the result, so we re-generate it for the response.
        # This is a quick operation as the image is already in memory.
        # from src.modules.image_caption import generate_caption
        # from PIL import Image
        # import base64, io
        # image_bytes = base64.b64decode(request.image_b64)
        # image = Image.open(io.BytesIO(image_bytes))
        # caption = await generate_caption(image)
        caption = "test"
        # Format the raw ChromaDB results into our Pydantic response model
        formatted_results = []
        ids = search_results.get('ids', [[]])[0]
        distances = search_results.get('distances', [[]])[0]
        metadatas = search_results.get('metadatas', [[]])[0]
        documents = search_results.get('documents', [[]])[0]

        for i, item_id in enumerate(ids):
            formatted_results.append(SearchResultItem(
                id=item_id,
                distance=distances[i],
                metadata=metadatas[i],
                document=documents[i]
            ))
        
        return ImageSearchResponse(results=formatted_results, generated_caption=caption, summary=summary)

    except Exception as e:
        logger.error(f"An unexpected error occurred in the search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_file(filename: str) -> bool:
    return filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    if not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")

    extension = image.filename.split('.')[-1]
    unique_filename = f"{uuid4().hex}.{extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as f:
        f.write(await image.read())

    with open(file_path, "rb") as image_file:
        image_data = image_file.read()

    base64_str = base64.b64encode(image_data).decode("utf-8")

    return UploadResponse(base64_str=base64_str)


@app.get("/health")
def health_check():
    """A simple health check endpoint to confirm the service is running."""
    return {"status": "ok" if service else "degraded"}
