##  Hosting with vLLM 






### You can run this with terminal 
```bash
docker run --runtime nvidia --gpus all \
    -v /fsx/ubuntu/fine-tune-qwen2-vl-with-llama-factory/models/qwen2_vl_7b_pissa_qlora_128_fintabnet_en/ :/opt/ml/Qwen2-VL-7B-QLoRA-Int4 \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.6.3  \
    --model /opt/ml/Qwen2-VL-7B-QLoRA-Int4 \
    --max-model-len 8192
```

