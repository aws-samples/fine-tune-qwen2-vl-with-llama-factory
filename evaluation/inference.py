import argparse
import datetime
import json
from pathlib import Path

from dataset import FinancialStatementDataset
from model import Qwen2VL
from tqdm import tqdm

PROMPT = 'Please generate accurate HTML code that represents the table structure shown in input image, including any merged cells.'


def evaluate(args):
    # Initialize model
    if args.model_name == 'qwen2_vl':
        model = Qwen2VL(model_path=args.model_path)
    else:
        raise ValueError(f'Unsupported model architecture: {args.model_name}')

    # Initialize dataset and DataLoader
    if args.dataset_name == 'financial_statement':
        dataset = FinancialStatementDataset()
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset_name}')

    # Create results directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(args.log_path) / args.model_name / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = results_dir / 'evaluation_log.txt'

    with open(log_file, 'w', encoding='utf-8') as f:
        for image_id, image_base64, cur_gt in tqdm(dataset, desc='Evaluating'):
            cur_pred = model(prompt=PROMPT, image_base64=image_base64)

            log_entry = {'id': int(image_id), 'pred': cur_pred, 'gt': cur_gt}
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')
            f.flush()  # Ensure the data is written immediately


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',
                        type=str,
                        choices=['qwen2_vl', 'internvl2', 'phi3_v'],
                        required=True,
                        help='Model architecture to use')
    parser.add_argument('--model-path',
                        type=str,
                        required=True,
                        help='Model path')
    parser.add_argument('--dataset-name',
                        type=str,
                        choices=['pubtabnet', 'financial_statement'],
                        default='financial_statement',
                        help='Eval dataset')
    parser.add_argument('--log-path',
                        type=str,
                        default='./logs',
                        help='Path to save log files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
