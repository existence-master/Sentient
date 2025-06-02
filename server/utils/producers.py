# Hypothetical Kafka Producer - integrate your actual Kafka library
# Example: in server/utils/kafka_producer.py
from kafka import KafkaProducer # Or from confluent_kafka import Producer
import json
import os

class KafkaProducerManager:
    _producer = None

    @staticmethod
    def get_producer():
        if KafkaProducerManager._producer is None:
            try:
                KafkaProducerManager._producer = KafkaProducer(
                    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(','),
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    # Add other Kafka configs: security, acks, retries, etc.
                )
                print("[KafkaProducer] Kafka Producer initialized.")
            except Exception as e:
                print(f"[KafkaProducer_ERROR] Failed to initialize Kafka Producer: {e}")
                # Decide how to handle this: raise error, or allow app to run without Kafka (log warnings)
        return KafkaProducerManager._producer

    @staticmethod
    async def send_message(topic, message_dict): # Make it async for motor
        producer = KafkaProducerManager.get_producer()
        if not producer:
            print(f"[KafkaProducer_ERROR] Cannot send message, producer not available. Topic: {topic}")
            return False
        try:
            # For kafka-python, send is synchronous by default but returns a Future.
            # To make it awaitable or handle async, you might need an asyncio loop executor
            # or use an async Kafka client if available.
            # For simplicity with standard kafka-python:
            loop = asyncio.get_event_loop()
            future = producer.send(topic, value=message_dict)
            await loop.run_in_executor(None, future.get, 60) # Timeout after 60s
            print(f"[KafkaProducer] Message sent to topic {topic}: {str(message_dict)[:100]}...")
            return True
        except Exception as e:
            print(f"[KafkaProducer_ERROR] Failed to send message to Kafka topic {topic}: {e}")
            return False