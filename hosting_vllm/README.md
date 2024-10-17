####  Evaluation using vLLM 

Once the model is merged, you can use vLLM for acceleration 
https://qwen.readthedocs.io/en/latest/deployment/vllm.html

```
# conda deactivate 
source ~/miniconda3/bin/activate
conda create -n vllm python=3.10 
conda activate vllm
pip install vllm
```

Error using vLLM (Qwen2 VL just merge, need to install vllm from source code)

  File "/fsx/ubuntu/miniconda3/envs/vllm/lib/python3.10/site-packages/vllm/config.py", line 1746, in _get_and_verify_max_len
    assert "factor" in rope_scaling
AssertionError