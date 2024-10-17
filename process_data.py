import argparse
import json
import multiprocessing as mp
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

PROMPT = '<image>Please generate accurate HTML code that represents the table structure shown in input image, including any merged cells.'


def th_to_td(html_string):
    html_string = re.sub(r'(?s)<th(.*?)>(.*?)</th>', r'<td\1>\2</td>',
                         html_string)
    return html_string


def remove_thead_tbody(html_string):
    html_string = re.sub(r'(?s)<thead.*?>(.*?)</thead>', r'\1', html_string)
    html_string = re.sub(r'(?s)<tbody.*?>(.*?)</tbody>', r'\1', html_string)
    return html_string


def remove_head(html_string):
    html_string = re.sub(r'(?s)<head.*?>(.*?)</head>', '', html_string)
    return html_string


def process_newlines_in_tr(html_content):

    def process_tr(match):
        tr_content = match.group(1)
        tr_content = re.sub(r'\s*(<[^>]+>)\s*', r'\1', tr_content)
        tr_content = re.sub(r'(</td>)(<td[^>]*?>)', r'\1\n\2', tr_content)
        tr_content = re.sub(r'(</th>)(<th[^>]*?>)', r'\1\n\2', tr_content)
        return f'<tr>\n{tr_content}\n</tr>'

    return re.sub(r'<tr>\s*(.*?)\s*</tr>',
                  process_tr,
                  html_content,
                  flags=re.DOTALL)


def clean_html_table(html_content):
    # Remove thead and tbody tags
    html_content = remove_thead_tbody(html_content)

    # Remove head tag
    html_content = remove_head(html_content)

    # th to td
    html_content = th_to_td(html_content)

    # Remove leading whitespace at the beginning of each line
    html_content = re.sub(r'^\s+', '', html_content, flags=re.MULTILINE)

    # Remove empty lines
    html_content = re.sub(r'\n\s*\n', '\n', html_content)

    # Remove trailing whitespace at the end of each line
    html_content = re.sub(r'\s+$', '', html_content, flags=re.MULTILINE)

    # Add html and body tags
    if not re.search(r'<html.*?>.*</html>', html_content,
                     re.DOTALL | re.IGNORECASE):
        if not re.search(r'<body.*?>.*</body>', html_content,
                         re.DOTALL | re.IGNORECASE):
            html_content = f'<html>\n<body>\n{html_content}\n</body>\n</html>'
        else:
            html_content = f'<html>\n{html_content}\n</html>'

    # process newlines in tr
    html_content = process_newlines_in_tr(html_content)
    return html_content.strip()


def process_sample(sample):
    messages = [{
        'role': 'user',
        'content': PROMPT,
    }, {
        'role': 'assistant',
        'content': clean_html_table(sample['html_table']),
    }]
    image_path = images_folder / f"{sample['imgid']}.jpg"
    sample['image'].save(image_path)

    return {'messages': messages, 'images': [str(image_path)]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process PubTabNet dataset')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./data/pubtabnet_v3',
                        help='Directory to store output files')
    parser.add_argument('--num_processes',
                        type=int,
                        default=32,
                        help='Number of parallel processes to use')

    args = parser.parse_args()

    #ds = load_dataset('apoidea/pubtabnet-html')['train']
    ds = load_dataset("apoidea/pubtabnet-html", split="train").select(range(4000))
    # ds = ds.remove_columns('image')

    data_folder = Path(args.output_dir)
    images_folder = data_folder / 'images'
    data_folder.mkdir(parents=True, exist_ok=True)
    images_folder.mkdir(parents=True, exist_ok=True)

    output_file = data_folder / 'pubtabnet.json'

    # Create a pool of worker processes
    with mp.Pool(processes=args.num_processes) as pool:
        # Process samples in parallel
        results = list(tqdm(pool.imap(process_sample, ds), total=len(ds)))

    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f'Processing complete. Output saved to {output_file}')