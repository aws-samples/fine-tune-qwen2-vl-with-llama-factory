from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

LOCAL_MODEL_PATH = './models/qwen2_vl_7b_pissa_qlora_128_fintabnet_en'
HuggingFace_MODEL_PATH = 'Qwen/Qwen2-VL-7B-Instruct'
IMAGE_PATHS = ["./evaluation/financial_table.png"]  # Add more image paths
max_pixels = 1500*1500


model = Qwen2VLForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)


# default processer
processor = AutoProcessor.from_pretrained(HuggingFace_MODEL_PATH,max_pixels=max_pixels)

# Prepare batch of messages
batch_messages = []
for image_path in IMAGE_PATHS:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": "Please generate accurate HTML code that represents the table structure shown in input image, including any merged cells."},
            ],
        }
    ]
    batch_messages.append(messages)

# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in batch_messages
]

image_inputs, _ = process_vision_info(batch_messages) 

inputs = processor(
    text=texts,
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")


# Set model to evaluation mode
model.eval()

# Batch Inference
# Batch inference: Generation of the output
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=4096)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)