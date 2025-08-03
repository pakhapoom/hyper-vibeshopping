import os
import json
import pika
import requests
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_rabbitmq_connection():
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
    return pika.BlockingConnection(parameters)

def call_ollama(endpoint, method, headers, body, query_params=None):
    """Make actual HTTP request to Ollama service"""
    ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
    ollama_port = os.getenv('OLLAMA_PORT', '11434')
    
    # Build URL
    url = f"http://{ollama_host}:{ollama_port}{endpoint}"
    if query_params:
        url += f"?{query_params}"
    
    # Clean headers (remove connection-specific headers)
    clean_headers = {
        key: value for key, value in headers.items()
        if key.lower() not in ['host', 'content-length', 'connection', 'accept-encoding']
    } if headers else {}
    
    try:
        logger.info(f"Making {method} request to {url}")
        logger.info(f"Request body to Ollama: {body}")
        logger.info(f"Request headers to Ollama: {clean_headers}")
        
        # Parse JSON body if Content-Type is application/json
        json_data = None
        if body and clean_headers.get('content-type', '').lower() == 'application/json':
            try:
                json_data = json.loads(body)
                body = None  # Use json parameter instead of data
                logger.info(f"Parsed JSON data: {json_data}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON body: {e}, using as string")
        
        # Make the request
        if method.upper() == 'GET':
            response = requests.get(url, headers=clean_headers, timeout=300)
        elif method.upper() == 'POST':
            if json_data:
                response = requests.post(url, json=json_data, headers=clean_headers, timeout=300)
            else:
                response = requests.post(url, data=body, headers=clean_headers, timeout=300)
        elif method.upper() == 'PUT':
            if json_data:
                response = requests.put(url, json=json_data, headers=clean_headers, timeout=300)
            else:
                response = requests.put(url, data=body, headers=clean_headers, timeout=300)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=clean_headers, timeout=300)
        elif method.upper() == 'PATCH':
            if json_data:
                response = requests.patch(url, json=json_data, headers=clean_headers, timeout=300)
            else:
                response = requests.patch(url, data=body, headers=clean_headers, timeout=300)
        else:
            if json_data:
                response = requests.request(method, url, json=json_data, headers=clean_headers, timeout=300)
            else:
                response = requests.request(method, url, data=body, headers=clean_headers, timeout=300)
        
        response.raise_for_status()
        
        # Try to parse as JSON, fallback to text
        try:
            response_data = response.json()
        except:
            response_data = {"response": response.text}
            
        return {
            "status": "success",
            "status_code": response.status_code,
            "response": response_data
        }
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Ollama {endpoint}: {e}")
        
        # Log response details if available
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Ollama response status: {e.response.status_code}")
            try:
                logger.error(f"Ollama response body: {e.response.text}")
            except:
                logger.error("Could not read response body")
        
        return {
            "status": "error",
            "status_code": 500,
            "error": str(e)
        }

def process_rpc_request(ch, method, properties, body):
    """Process RPC request and send response back"""
    try:
        request_data = json.loads(body)
        
        endpoint = request_data.get('endpoint', '/')
        http_method = request_data.get('method', 'GET')
        headers = request_data.get('headers', {})
        request_body = request_data.get('body')
        query_params = request_data.get('query_params')
        
        logger.info(f"Processing RPC request: {http_method} {endpoint}")
        logger.info(f"Received from RabbitMQ: {json.dumps(request_data, indent=2)}")
        
        # Call Ollama service
        result = call_ollama(endpoint, http_method, headers, request_body, query_params)
        
        logger.info(f"Ollama response status: {result.get('status')}")
        
        # Send response back to the reply_to queue
        ch.basic_publish(
            exchange='',
            routing_key=properties.reply_to,
            properties=pika.BasicProperties(
                correlation_id=properties.correlation_id
            ),
            body=json.dumps(result)
        )
        
        # Acknowledge the message
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
        logger.info(f"RPC response sent with correlation_id: {properties.correlation_id}")
        
    except Exception as e:
        logger.error(f"Error processing RPC request: {e}")
        
        # Send error response
        error_response = {
            "status": "error",
            "status_code": 500,
            "error": str(e)
        }
        
        try:
            ch.basic_publish(
                exchange='',
                routing_key=properties.reply_to,
                properties=pika.BasicProperties(
                    correlation_id=properties.correlation_id
                ),
                body=json.dumps(error_response)
            )
        except Exception as send_error:
            logger.error(f"Failed to send error response: {send_error}")
        
        # Acknowledge the message even on error to prevent redelivery
        ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    logger.info("Starting Hacktron Example RPC Consumer")
    
    # Wait for RabbitMQ to be ready
    while True:
        try:
            connection = get_rabbitmq_connection()
            break
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            logger.info("Retrying in 5 seconds...")
            time.sleep(5)
    
    try:
        channel = connection.channel()
        
        # Declare the RPC queue
        channel.queue_declare(queue='rpc_queue', durable=True)
        
        # Set QoS to process one message at a time
        channel.basic_qos(prefetch_count=1)
        
        # Set up RPC consumer
        channel.basic_consume(
            queue='rpc_queue',
            on_message_callback=process_rpc_request
        )
        
        logger.info("RPC Consumer is waiting for requests. To exit press CTRL+C")
        channel.start_consuming()
        
    except KeyboardInterrupt:
        logger.info("RPC Consumer interrupted by user")
        channel.stop_consuming()
        connection.close()
    except Exception as e:
        logger.error(f"RPC Consumer error: {e}")
    finally:
        if connection and not connection.is_closed:
            connection.close()

if __name__ == "__main__":
    main()
