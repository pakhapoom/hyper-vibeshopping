import os
import json
import logging
import requests
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from src.service import multimodal_search_service
from src.modules.login import login
from src.rpc import RPCClient


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
rpc_client = RPCClient()


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
        search_results, summary = await service.search_by_image(
            top_k=request.top_k, 
            user_input= request.user_input, 
            cust_info=request.cust_info,
        )
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
        # caption = "test"
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

def get_ollama_url():
    """Get Ollama URL for direct streaming requests"""
    ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
    ollama_port = os.getenv('OLLAMA_PORT', '11434')
    return f"http://{ollama_host}:{ollama_port}"

def is_streaming_request(request_body):
    """Check if request contains stream=true parameter"""
    if not request_body:
        return False
    try:
        data = json.loads(request_body)
        return data.get('stream', False) is True
    except:
        return False

async def proxy_streaming_request(endpoint, method, headers, body, query_params=None):
    """Direct proxy for streaming requests (bypass RabbitMQ)"""
    ollama_url = get_ollama_url()
    url = f"{ollama_url}{endpoint}"
    if query_params:
        url += f"?{query_params}"
    
    # Clean headers
    clean_headers = {
        key: value for key, value in headers.items()
        if key.lower() not in ['host', 'content-length', 'connection', 'accept-encoding']
    } if headers else {}
    
    logger.info(f"Direct streaming proxy: {method} {url}")
    logger.info(f"Streaming request body: {body}")
    logger.info(f"Streaming request headers: {clean_headers}")
    
    # Parse JSON body if Content-Type is application/json
    json_data = None
    if body and clean_headers.get('content-type', '').lower() == 'application/json':
        try:
            json_data = json.loads(body)
            body = None  # Use json parameter instead of data
            logger.info(f"Parsed streaming JSON data: {json_data}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse streaming JSON body: {e}, using as string")
    
    try:
        if json_data:
            response = requests.request(
                method=method,
                url=url,
                json=json_data,
                headers=clean_headers,
                stream=True,  # Enable streaming
                timeout=300
            )
        else:
            response = requests.request(
                method=method,
                url=url,
                data=body,
                headers=clean_headers,
                stream=True,  # Enable streaming
                timeout=300
            )
        response.raise_for_status()
        
        def generate():
            try:
                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    if chunk:
                        yield chunk
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                response.close()
        
        # Return streaming response with appropriate content type
        content_type = response.headers.get('content-type', 'application/json')
        return StreamingResponse(
            generate(),
            media_type=content_type,
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Streaming proxy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.api_route("/v1/models", methods=["GET"])
async def proxy_v1_models():
    """Proxy /v1/models requests via RabbitMQ RPC"""
    try:
        request_data = {
            "endpoint": "/v1/models",
            "method": "GET",
            "headers": {},
            "body": None,
            "timestamp": str(datetime.utcnow())
        }
        
        logger.info("Proxying GET /v1/models via RPC")
        result = rpc_client.call_rpc(request_data)
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
            
        return JSONResponse(
            content=result.get('response'),
            status_code=result.get('status_code', 200),
            headers={"Content-Type": "application/json"}
        )
        
    except Exception as e:
        logger.error(f"Error proxying /v1/models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/v1/chat/completions", methods=["POST"])
async def proxy_v1_chat_completions(request: Request):
    """Proxy /v1/chat/completions requests via RabbitMQ RPC or direct streaming"""
    try:
        body = await request.body()
        headers = dict(request.headers)
        body_str = body.decode() if body else None
        
        # Check if this is a streaming request
        if is_streaming_request(body_str):
            logger.info("Detected streaming request, using direct proxy")
            return await proxy_streaming_request("/v1/chat/completions", "POST", headers, body_str)
        
        # Non-streaming request - use RPC
        request_data = {
            "endpoint": "/v1/chat/completions",
            "method": "POST",
            "headers": headers,
            "body": body_str,
            "timestamp": str(datetime.utcnow())
        }
        
        logger.info("Proxying POST /v1/chat/completions via RPC")
        logger.info(f"Sending to RabbitMQ: {json.dumps(request_data, indent=2)}")
        
        try:
            result = rpc_client.call_rpc(request_data)
            logger.debug(f"RPC result: {result}")
            
            if result.get('status') == 'error':
                error_detail = result.get('error', 'Unknown error')
                logger.error(f"RPC returned error: {error_detail}")
                raise HTTPException(status_code=500, detail=error_detail)
                
            return JSONResponse(
                content=result.get('response'),
                status_code=result.get('status_code', 200),
                headers={"Content-Type": "application/json"}
            )
            
        except Exception as rpc_error:
            logger.error(f"RPC call failed: {rpc_error}")
            raise HTTPException(status_code=500, detail=f"RPC error: {str(rpc_error)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /v1/chat/completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.api_route("/api/tags", methods=["GET"])
async def proxy_api_tags():
    """Proxy /api/tags requests via RabbitMQ RPC"""
    try:
        request_data = {
            "endpoint": "/api/tags",
            "method": "GET",
            "headers": {},
            "body": None,
            "timestamp": str(datetime.utcnow())
        }
        
        logger.info("Proxying GET /api/tags via RPC")
        result = rpc_client.call_rpc(request_data)
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
            
        return JSONResponse(
            content=result.get('response'),
            status_code=result.get('status_code', 200),
            headers={"Content-Type": "application/json"}
        )
        
    except Exception as e:
        logger.error(f"Error proxying /api/tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/api/generate", methods=["POST"])
async def proxy_api_generate(request: Request):
    """Proxy /api/generate requests via RabbitMQ RPC or direct streaming"""
    try:
        body = await request.body()
        headers = dict(request.headers)
        body_str = body.decode() if body else None
        
        # Check if this is a streaming request
        if is_streaming_request(body_str):
            logger.info("Detected streaming request, using direct proxy")
            return await proxy_streaming_request("/api/generate", "POST", headers, body_str)
        
        # Non-streaming request - use RPC
        request_data = {
            "endpoint": "/api/generate",
            "method": "POST",
            "headers": headers,
            "body": body_str,
            "timestamp": str(datetime.utcnow())
        }
        
        logger.info("Proxying POST /api/generate via RPC")
        result = rpc_client.call_rpc(request_data)
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
            
        return JSONResponse(
            content=result.get('response'),
            status_code=result.get('status_code', 200),
            headers={"Content-Type": "application/json"}
        )
        
    except Exception as e:
        logger.error(f"Error proxying /api/generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Catch-all proxy for other endpoints
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_all(request: Request, path: str):
    """Proxy all other requests via RabbitMQ RPC"""
    try:
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            body_bytes = await request.body()
            body = body_bytes.decode() if body_bytes else None
            
        headers = dict(request.headers)
        query_params = str(request.url.query) if request.url.query else None
        
        request_data = {
            "endpoint": f"/{path}",
            "method": request.method,
            "headers": headers,
            "body": body,
            "query_params": query_params,
            "timestamp": str(datetime.utcnow())
        }
        
        logger.info(f"Proxying {request.method} /{path} via RPC")
        result = rpc_client.call_rpc(request_data)
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
            
        return JSONResponse(
            content=result.get('response'),
            status_code=result.get('status_code', 200),
            headers={"Content-Type": "application/json"}
        )
        
    except Exception as e:
        logger.error(f"Error proxying /{path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))