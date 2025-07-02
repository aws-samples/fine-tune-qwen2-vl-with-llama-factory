from locust import HttpUser, task, between, events
import json
import random
import string
import time
import requests



class VLLMBenchmark(HttpUser):
    host = "http://localhost:8080"
    wait_time = between(0.1, 0.2)

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

        # Randomly insert into blog content
        insert_pos = random.randint(0, len(self.blog_content))
        modified_content = (
            self.blog_content[:insert_pos] +
            ' ' + random_words + ' ' +
            self.blog_content[insert_pos:]
        )

        # Build payload
        payload = {
            "model": "Qwen/Qwen3-30B-A3B",
            #"model": "Qwen/Qwen3-235B-A22B",
            "messages": [{"role": "user", "content": modified_content + "Can you please write in a clear way about 1500 words"}],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 1000,
            "presence_penalty": 1.5,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False
        }

        # Track request manually
        start_time = time.time()
        full_url = f"{self.host}/v1/chat/completions"

        try:
            # Make the HTTP POST request
            response = requests.post(full_url, json=payload, timeout=180.0)
            response.raise_for_status()  # Raise for 4xx/5xx status codes

            # Parse response
            response_body = response.json()

            # Calculate latency
            total_time_ms = (time.time() - start_time) * 1000

            # Extract token usage
            usage = response_body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Fire success event with metrics
            self.environment.events.request.fire(
                request_type="HTTP",
                name="chat_completion",
                response_time=total_time_ms,
                response_length=len(response.content),
                exception=None,
                context={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            )

        except requests.exceptions.RequestException as e:
            # Handle HTTP-level errors
            total_time_ms = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="HTTP",
                name="chat_completion",
                response_time=total_time_ms,
                exception=e
            )

        except json.JSONDecodeError:
            # Handle JSON parsing errors
            total_time_ms = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="HTTP",
                name="chat_completion",
                response_time=total_time_ms,
                exception=ValueError("Failed to decode JSON response")
            )