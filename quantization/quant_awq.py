import argparse
import json

from awq.models.qwen2vl import Qwen2VLAWQForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor


def prepare_dataset(file_path: str, n_sample: int = 8) -> list[list[dict]]:
    dataset = []
    with open(file_path, encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i >= n_sample:
                break
            data = json.loads(line)
            user_message = data['messages'][0]
            assistant_message = data['messages'][1]
            image_path = data['images'][0]

            dataset.append([{
                'role':
                'user',
                'content': [{
                    'type': 'image',
                    'image': image_path
                }, {
                    'type':
                    'text',
                    'text':
                    user_message['content'].replace('<image>', '')
                }],
            }, {
                'role': 'assistant',
                'content': assistant_message['content']
            }])
    return dataset


def main(args):
    # Load your processor and model with AutoAWQ
    processor = Qwen2VLProcessor.from_pretrained(args.model_path)

    device_map = {
        'visual': 0,
        'model.embed_tokens': 0,
        'model.rotary_emb': 0,
        'model.layers.0': 0,
        'model.layers.1': 1,
        'model.layers.2': 1,
        'model.layers.3': 1,
        'model.layers.4': 2,
        'model.layers.5': 2,
        'model.layers.6': 2,
        'model.layers.7': 2,
        'model.layers.8': 2,
        'model.layers.9': 3,
        'model.layers.10': 3,
        'model.layers.11': 3,
        'model.layers.12': 3,
        'model.layers.13': 4,
        'model.layers.14': 4,
        'model.layers.15': 4,
        'model.layers.16': 4,
        'model.layers.17': 5,
        'model.layers.18': 5,
        'model.layers.19': 5,
        'model.layers.20': 5,
        'model.layers.21': 6,
        'model.layers.22': 6,
        'model.layers.23': 6,
        'model.layers.24': 6,
        'model.layers.25': 7,
        'model.layers.26': 7,
        'model.layers.27': 7,
        'model.norm': 7,
        'lm_head': 7,
    }
    model = Qwen2VLAWQForConditionalGeneration.from_pretrained(
        args.model_path,
        model_type='qwen2_vl',
        use_cache=False,
        device_map=device_map,
        attn_implementation='flash_attention_2')

    # Prepare dataset
    dataset = prepare_dataset(args.jsonl_file, args.n_sample)

    # Process the dataset
    text = processor.apply_chat_template(dataset,
                                         tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(dataset)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    )

    # Quantization config
    quant_config = {
        'zero_point': True,
        'q_group_size': args.q_group_size,
        'w_bit': args.w_bit,
        'version': 'GEMM'
    }

    # Quantize the model
    model.quantize(calib_data=inputs, quant_config=quant_config)
    model.model.config.use_cache = model.model.generation_config.use_cache = True

    # Save the quantized model
    model.save_quantized(args.quant_path, safetensors=True, shard_size='2GB')
    processor.save_pretrained(args.quant_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantize Qwen2VL model')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to the original model')
    parser.add_argument('--quant_path',
                        type=str,
                        required=True,
                        help='Path to save the quantized model')
    parser.add_argument('--jsonl_file',
                        type=str,
                        required=True,
                        help='Path to the JSONL file')
    parser.add_argument('--n_sample',
                        type=int,
                        default=16,
                        help='Number of samples to use')
    parser.add_argument('--q_group_size',
                        type=int,
                        default=128,
                        help='Quantization group size')
    parser.add_argument('--w_bit',
                        type=int,
                        default=4,
                        help='Weight bit width for quantization')

    args = parser.parse_args()
    main(args)