import os
import json
import pika
import uvicorn
import uuid
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hacktron Example RPC Proxy", version="1.0.0")

class RPCClient:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.callback_queue = None
        self.responses = {}
        self.correlation_id = None
        
    def connect(self):
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
            blocked_connection_timeout=300
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        
        # Declare callback queue for RPC responses
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue
        
        # Declare RPC request queue
        self.channel.queue_declare(queue='rpc_queue', durable=True)
        
        # Set up consumer for callback queue
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )
        
    def on_response(self, ch, method, props, body):
        if self.correlation_id == props.correlation_id:
            self.responses[props.correlation_id] = body
            
    def call_rpc(self, request_data):
        try:
            if not self.connection or self.connection.is_closed:
                logger.info("RPC connection closed, reconnecting...")
                self.connect()
                
            self.correlation_id = str(uuid.uuid4())
            
            logger.debug(f"Sending RPC request with correlation_id: {self.correlation_id}")
            
            # Send RPC request
            self.channel.basic_publish(
                exchange='',
                routing_key='rpc_queue',
                properties=pika.BasicProperties(
                    reply_to=self.callback_queue,
                    correlation_id=self.correlation_id,
                    delivery_mode=2
                ),
                body=json.dumps(request_data)
            )
            
            # Wait for response with timeout
            timeout = 300  # 5 minutes
            start_time = datetime.now()
            
            while self.correlation_id not in self.responses:
                self.connection.process_data_events(time_limit=1)
                elapsed = (datetime.now() - start_time).seconds
                if elapsed > timeout:
                    raise Exception(f"RPC call timeout after {elapsed} seconds")
                    
            response = self.responses.pop(self.correlation_id)
            logger.debug(f"Received RPC response for correlation_id: {self.correlation_id}")
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"RPC call failed: {str(e)}")
            raise
        
    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()

# Global RPC client instance
rpc_client = RPCClient()

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

@app.get("/")
async def root():
    return {"message": "Hacktron Example RPC Proxy is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rpc-proxy"}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
