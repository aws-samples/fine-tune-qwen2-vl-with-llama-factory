from openai import OpenAI
import requests
import json
import base64
import argparse
import os
from pathlib import Path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze images using Qwen2-VL model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--image',
        required=True,
        type=str,
        help='Path to the image file'
    )
    parser.add_argument(
        '--model_path',
        default="/opt/ml/Qwen2-VL-7B-QLoRA-Int4",
        type=str,
        help='Path to the Qwen2-VL model'
    )
    return parser.parse_args()

def validate_image_path(image_path):
    """Validate if the image path exists and is a valid image file"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # List of valid image extensions
    valid_extensions = {'.png'}
    file_extension = Path(image_path).suffix.lower()
    
    if file_extension not in valid_extensions:
        raise ValueError(f"Invalid image format. Supported formats: {', '.join(valid_extensions)}")
    
    return True

def encode_image(image_path):
    """Encode image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise Exception(f"Error encoding image: {str(e)}")

def analyze_image(image_path, model_path):
    """Analyze image using the specified model"""
    # Initialize OpenAI client with custom settings
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

    # Encode image to base64
    base64_image = encode_image(image_path)

    # Prepare the messages for the API request
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please generate accurate HTML code that represents the table structure shown in input image, including any merged cells."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    # Make the API request
    try:
        completion = client.chat.completions.create(
            model=model_path,
            messages=messages,
            max_tokens=4096
        )
        return completion
    except Exception as e:
        return f"Error occurred: {str(e)}"

def main():
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Validate image path
        validate_image_path(args.image)
        
        # Analyze image
        print(f"Analyzing image: {args.image}")
        result = analyze_image(args.image, args.model_path)
        print("\nCompletion result:", result)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()