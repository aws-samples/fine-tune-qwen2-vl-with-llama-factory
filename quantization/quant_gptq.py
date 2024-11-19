import argparse
import json
import logging

from auto_gptq import BaseQuantizeConfig
from auto_gptq.modeling import Qwen2VLGPTQForConditionalGeneration
from optimum.gptq import GPTQQuantizer
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor
from transformers.utils.quantization_config import QuantizationMethod

logging.basicConfig(format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


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


def batched(iterable, n: int):
    # batched('ABCDEFG', 3) → ABC DEF G
    assert n >= 1, 'batch size must be at least one'
    from itertools import islice

    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def main(args):
    # Load your processor and model with AutoAWQ
    processor = Qwen2VLProcessor.from_pretrained(args.model_path)

    quantize_config = BaseQuantizeConfig(
        bits=args.w_bit,  # 4 or 8
        group_size=args.q_group_size,
        damp_percent=0.1,
        desc_act=False,
        static_groups=False,
        sym=True,
        true_sequential=True,
    )

    model = Qwen2VLGPTQForConditionalGeneration.from_pretrained(
        args.model_path,
        quantize_config,
        attn_implementation='flash_attention_2')

    # Prepare dataset
    dataset = prepare_dataset(args.jsonl_file, args.n_sample)

    batch_size = 1
    calib_data = []
    for batch in batched(dataset, batch_size):
        text = processor.apply_chat_template(batch,
                                             tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(batch)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        )
        calib_data.append(inputs)

    model.quantize(calib_data, cache_examples_on_gpu=True)

    # Save the quantized model
    gptq_quantizer = GPTQQuantizer.from_dict(model.quantize_config.to_dict())
    model.model.is_quantized = True
    model.model.quantization_method = QuantizationMethod.GPTQ
    model.model.config.quantization_config = gptq_quantizer.to_dict()
    model.model.save_pretrained(args.quant_path, max_shard_size='4GB')
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
                        default=92,
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
