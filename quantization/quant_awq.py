import argparse
import json

from awq.models.qwen2vl import Qwen2VLAWQForCausalLM

from awq import AutoAWQForCausalLM

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from awq.quantize.quantizer import AwqQuantizer, clear_memory, get_best_device

import torch
import torch.nn as nn

class Qwen2VLAwqQuantizer(AwqQuantizer):
    def init_quant(self, n_samples=None, max_seq_len=None):
        modules = self.awq_model.get_model_layers(self.model)
        samples = self.calib_data

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        def move_to_device(obj: torch.Tensor | nn.Module, device: torch.device):
            def get_device(obj: torch.Tensor | nn.Module):
                if isinstance(obj, torch.Tensor):
                    return obj.device
                return next(obj.parameters()).device

            if get_device(obj) != device:
                obj = obj.to(device)
            return obj

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        for k, v in samples.items():
            if isinstance(v, (torch.Tensor, nn.Module)):
                samples[k] = move_to_device(v, best_device)
        try:
            self.model(**samples)
        except ValueError:  # work with early exit
            pass
        finally:
            for k, v in samples.items():
                if isinstance(v, (torch.Tensor, nn.Module)):
                    samples[k] = move_to_device(v, "cpu")
        modules[0] = modules[0].module  # restore

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        return modules, layer_kwargs, inps
    
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
    max_pixels = 1500*1500

    # Load your processor and model with AutoAWQ
    processor = AutoProcessor.from_pretrained(args.model_path,
                                              chat_template=("{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}" 
                                                            ),
                                            max_pixels=max_pixels
                                            )

    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
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
    model.quantize(calib_data=inputs, quant_config=quant_config, quantizer_cls=Qwen2VLAwqQuantizer)
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
