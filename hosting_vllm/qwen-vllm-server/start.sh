#!/bin/bash

# Start vLLM server in the background
vllm serve Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4 \
    --max-model-len 8192 \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 &


# Wait for vLLM server to start
echo "Waiting for vLLM server to be ready..."

# Check chat completions endpoint instead
until curl -s -f "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", "messages": [{"role": "user", "content": "hi"}]}' > /dev/null 2>&1; do
    echo "Waiting for vLLM server..."
    sleep 5
done

echo "vLLM server is ready!"
echo "Starting FastAPI server..."
python app.py