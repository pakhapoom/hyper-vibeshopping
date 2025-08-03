import os
import json
import pika
import uvicorn
import uuid
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from src.service import multimodal_search_service
from src.modules.login import login


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class ImageSearchRequest(BaseModel):
    """Defines the structure of the incoming API request."""
    user_input: str = Field(..., description="User's input text for search context.")
    cust_info: List[Dict[str, Any]] = Field(..., description="Customer information for personalized search.")
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
    # generated_caption: str
    summary: str

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    authentication: bool
    cust_info: List[Dict[str, Any]]
    
class UploadRequest(BaseModel):
    image: UploadFile = None

class UploadResponse(BaseModel):
    file_path: str


class RPCClient:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.callback_queue = None
        self.responses = {}
        self.correlation_id = None
        self.connect()
        
    def connect(self, queue_name="rpc_queue"):
        rabbitmq_host = os.getenv('RABBITMQ_HOST', 'localhost')
        rabbitmq_port = int(os.getenv('RABBITMQ_PORT', 5672))
        rabbitmq_user = os.getenv('RABBITMQ_USER', 'admin')
        rabbitmq_pass = os.getenv('RABBITMQ_PASS', 'password123')
        
        credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_pass)
        parameters = pika.ConnectionParameters(
            host=rabbitmq_host,
            port=rabbitmq_port,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300,
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=queue_name)
    
    def on_request(self, ch, method, props, body):
        response = search_by_image_endpoint(
            user_input=body.user_input,
            cust_info=body.cust_info,
            top_k=body.top_k,
        )
 
        ch.basic_publish(exchange='',
                        routing_key=props.reply_to,
                        properties=pika.BasicProperties(correlation_id = \
                                                            props.correlation_id),
                        body=str(response))
        ch.basic_ack(delivery_tag=method.delivery_tag)

 
rpc_client = RPCClient()
rpc_client.channel.basic_qos(prefetch_count=1)
rpc_client.channel.basic_consume(queue='rpc_queue', on_message_callback=on_request)
 
print(" [x] Awaiting RPC requests")
rpc_client.channel.start_consuming()

@app.on_event("startup")
async def startup_event():
    try:
        # Try to connect with retries
        max_retries = 5
        for attempt in range(max_retries):
            try:
                rpc_client.connect()
                logger.info("Successfully connected to RabbitMQ RPC system")
                break
            except Exception as e:
                logger.warning(f"RabbitMQ connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("Failed to connect to RabbitMQ after all retries")
                else:
                    import time
                    time.sleep(2)
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    try:
        rpc_client.close()
        logger.info("RPC client connection closed")
    except Exception as e:
        logger.error(f"Error closing RPC client: {e}")


# HyperVibe endpoints
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
        search_results, summary = await service.search_by_image(
            top_k=request.top_k, 
            user_input= request.user_input, 
            cust_info=request.cust_info,
        )
        print(f"Search results: {search_results}")
        if search_results is None:
            raise HTTPException(status_code=404, detail="Could not find any results. The image might not have generated a valid caption.")

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
        
        return ImageSearchResponse(results=formatted_results, summary=summary)

    except Exception as e:
        logger.error(f"An unexpected error occurred in the search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_file(filename: str) -> bool:
    return filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS

@app.post("/upload")
async def upload_image(image: UploadFile = File(None)): # File(...)
    if image is None:
        raise HTTPException(status_code=400, detail="No image uploaded")

    if not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")

    unique_filename = "temp_img.png"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as f:
        f.write(await image.read())

    return UploadResponse(file_path=file_path)


@app.get("/health")
def health_check():
    """A simple health check endpoint to confirm the service is running."""
    return {"status": "ok" if service else "degraded"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
