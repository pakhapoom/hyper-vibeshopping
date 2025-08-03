import os
import json
import pika
import requests
import logging
import time
from datetime import datetime
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageSearchRequest(BaseModel):
    """Defines the structure of the incoming API request."""
    user_input: str = Field(..., description="User's input text for search context.")
    cust_info: List[Dict[str, Any]] = Field(..., description="Customer information for personalized search.")
    top_k: int = Field(5, gt=0, le=20, description="The number of top results to return.")


class ChatRpcClient(object):

    def __init__(self):
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
        result = self.channel.queue_declare(queue=queue_name, exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, user_input, cust_info, top_k):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=ImageSearchRequest(
                user_input=user_input,
                cust_info=cust_info,
                top_k=top_k
            ))
        while self.response is None:
            self.connection.process_data_events(time_limit=None)
        return int(self.response)


def main():
    logger.info("Starting Hacktron Example RPC Consumer")
    try:
        chat_rpc = ChatRpcClient()
        channel = chat_rpc.connection.channel()
        
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
        chat_rpc.connection.close()
    except Exception as e:
        logger.error(f"RPC Consumer error: {e}")
    finally:
        if chat_rpc.connection and not chat_rpc.connection.is_closed:
            chat_rpc.connection.close()


if __name__ == "__main__":
    main()
