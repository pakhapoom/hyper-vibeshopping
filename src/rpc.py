import os
import json
import pika
import uuid
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
