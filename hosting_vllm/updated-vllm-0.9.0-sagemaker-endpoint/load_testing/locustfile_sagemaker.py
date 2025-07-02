from locust import HttpUser, task, between
import json
import random
import string
import boto3
import time

# SageMaker configuration
endpoint_name = "Qwen3-30B-2025-06-12-10-58-48-818"  # Replace with your endpoint name
REGION_NAME = "us-east-1"  # AWS region

# Initialize SageMaker runtime client
runtime_sm_client = boto3.client('sagemaker-runtime', region_name=REGION_NAME)

class SageMakerBenchmark(HttpUser):
    host = "http://localhost"  # Dummy host to satisfy Locust
    wait_time = between(0.1, 0.2)  # Simulated user think time

    # Read blog content once at class load
    try:
        with open("blog", "r", encoding="utf-8") as f:
            blog_content = f.read()
    except Exception as e:
        blog_content = "Blog content not found or unreadable."
        print(f"Error reading blog file: {e}")

    @task
    def chat_completion(self):
        # Generate pseudo-random words
        random_words = ' '.join(
            ''.join(random.choices(string.ascii_lowercase, k=5)) 
            for _ in range(10)
        )

        # Insert into blog content randomly
        insert_pos = random.randint(0, len(self.blog_content))
        modified_content = (
            self.blog_content[:insert_pos] +
            ' ' + random_words + ' ' +
            self.blog_content[insert_pos:]
        )

        # Build payload
        payload = {
            #"model": "Qwen/Qwen3-30B-A3B",
            "model": "Qwen/Qwen3-235B-A22B",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": modified_content + "Can you please write in a clear way about 1500 words"}
                    ]
                }
            ],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 1000,
            "presence_penalty": 1.5,
            "stream": False
        }

        # Track request manually
        start_time = time.time()
        try:
            response = runtime_sm_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            # Parse body
            response_body = json.loads(response['Body'].read().decode())

            # Calculate latency
            total_time = (time.time() - start_time) * 1000  # in milliseconds

            # Extract token usage
            usage = response_body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Fire success event for metrics
            self.environment.events.request.fire(
                request_type="boto3",
                name="sagemaker_invoke",
                response_time=total_time,
                response_length=len(json.dumps(response_body)),
                exception=None,
                context={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            )

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            # Fire failure event
            self.environment.events.request.fire(
                request_type="boto3",
                name="sagemaker_invoke",
                response_time=total_time,
                exception=e
            )