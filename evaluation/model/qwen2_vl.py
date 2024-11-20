import torch.nn as nn
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch 

class Qwen2VL(nn.Module):

    def __init__(self,
                 model_path='Qwen/Qwen2-VL-7B-Instruct',
                 device_map='auto',
                 torch_dtype = torch.bfloat16,
                 use_flash_attention_2=True):
        super().__init__()
        if use_flash_attention_2:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation='flash_attention_2',
                device_map=device_map,
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch_dtype, device_map=device_map)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            chat_template=(
                "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"  # noqa: E501
            ))

    def forward(self,
                prompt,
                image_base64,
                max_new_tokens=8192,
                repetition_penalty=1.05,
                do_sample=True,
                temperature=1.0):
        messages = [{
            'role':
            'user',
            'content': [
                {
                    'type': 'image',
                    'image': f'data:image;base64,{image_base64}'
                },
                {
                    'type': 'text',
                    'text': prompt
                },
            ]
        }]

        texts = [
            self.processor.apply_chat_template(messages,
                                               tokenize=False,
                                               add_generation_prompt=True)
        ]
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            temperature=temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return output_texts[0]
