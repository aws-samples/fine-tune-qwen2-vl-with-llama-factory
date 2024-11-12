from fastapi import FastAPI, HTTPException
from openai import OpenAI
from openai import AsyncOpenAI  # Note the AsyncOpenAI import

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import json


# Adapt your own inference container for Amazon SageMaker
# https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html

app = FastAPI()


# vLLM client configuration
VLLM_CLIENT = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

class GenerateRequest(BaseModel):
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = 1024

@app.post("/invocations")
async def generate_text(request: GenerateRequest):
    print("Received request")
    try:
        # Use await with the async client
        response = await VLLM_CLIENT.chat.completions.create(
            model="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
            messages=request.messages,
            max_tokens=request.max_tokens
        )
        return json.dumps({"response": response.choices[0].message.content})  
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")        

    
@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)