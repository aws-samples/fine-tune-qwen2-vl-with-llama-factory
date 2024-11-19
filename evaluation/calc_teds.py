import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import tqdm
from metric import TEDS


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


def remove_spaces_in_numbers(html_content):
    pattern = r'(<td[^>]*>)(.*?)(</td>)'

    def replace_function(match):
        opening_tag = match.group(1)
        content = match.group(2)
        closing_tag = match.group(3)
        content = re.sub(r'(\d{1,3}(?:,\s*\d{3})*(?:\.\d+)?)',
                         lambda m: m.group(1).replace(' ', ''), content)

        return f'{opening_tag}{content}{closing_tag}'

    return re.sub(pattern, replace_function, html_content, flags=re.DOTALL)


def convert_to_colspan(html_content):
    pattern = r'(<td>[^<]*</td>)(\s*<td>\s*</td>)*'

    def replace_func(match):
        content_td = match.group(1)
        empty_td_count = len(re.findall(r'<td>\s*</td>', match.group(0)))
        colspan = empty_td_count + 1
        if colspan > 1:
            content = f'<td colspan="{colspan}">{content_td[4:-5]}</td>'
        else:
            content = match.group(0)
        return content

    return re.sub(pattern, replace_func, html_content)


def clean_html_table(html_content, is_gt=False):
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

    if not is_gt:
        # Remove spaces in number, e.g., 123,456,789
        html_content = remove_spaces_in_numbers(html_content)
    # print(html_content)
    return html_content.strip()


def load_jsonl(file_path):
    with open(file_path) as f:
        return [json.loads(line) for line in f]


def save_html_files(entry, output_dir):
    id = entry['id']
    gt_path = os.path.join(output_dir, f'{id}_gt.html')
    pred_path = os.path.join(output_dir, f'{id}_pred.html')

    with open(gt_path, 'w', encoding='utf-8') as f:
        print(entry['gt'], file=f)

    with open(pred_path, 'w', encoding='utf-8') as f:
        print(entry['pred'], file=f)


def calculate_score(entry, output_dir):
    teds_structure = TEDS(structure_only=True)
    teds = TEDS(structure_only=False)
    pred = clean_html_table(entry['pred'], is_gt=False)
    gt = clean_html_table(entry['gt'], is_gt=True)

    save_html_files(entry, output_dir)

    try:
        teds_score = teds.evaluate(pred, gt)
    except Exception:
        print(f"Error {entry['id']}")
        teds_score = 0.
    try:
        teds_structure_score = teds_structure.evaluate(pred, gt)
    except Exception:
        print(f"Error {entry['id']}")
        teds_structure_score = 0.

    return entry['id'], teds_score, teds_structure_score


def calculate_scores(data, num_workers, output_dir):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        scores = list(
            tqdm.tqdm(executor.map(calculate_score, data,
                                   [output_dir] * len(data)),
                      total=len(data),
                      desc='Calculating TEDS scores'))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Path to the JSONL file')
    parser.add_argument('--num_workers',
                        type=int,
                        default=16,
                        help='Number of worker processes (default: 16)')
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(args.file_path), 'htmls')
    os.makedirs(output_dir, exist_ok=True)
    data = load_jsonl(args.file_path)
    scores = calculate_scores(data, args.num_workers, output_dir)
    scores_with_diff = [(id, teds, teds_structure, teds_structure - teds)
                        for id, teds, teds_structure in scores]

    # Sort by TEDS score in ascending order
    scores_sorted_by_teds = sorted(scores_with_diff, key=lambda x: x[1])

    # Sort by difference in descending order
    scores_sorted_by_diff = sorted(scores_with_diff,
                                   key=lambda x: x[3],
                                   reverse=True)

    average_score_teds = sum(s[1]
                             for s in scores_with_diff) / len(scores_with_diff)
    average_scores_teds_structure = sum(
        s[2] for s in scores_with_diff) / len(scores_with_diff)

    print('\n10 samples with lowest TEDS scores:')
    for id, teds, teds_structure, diff in scores_sorted_by_teds[:10]:
        print(
            f'ID: {id}, TEDS: {teds:.4f}, TEDS structure: {teds_structure:.4f}, Difference: {diff:.4f}'
        )

    print('\n10 samples with largest difference (TEDS structure - TEDS):')
    for id, teds, teds_structure, diff in scores_sorted_by_diff[:10]:
        print(
            f'ID: {id}, TEDS: {teds:.4f}, TEDS structure: {teds_structure:.4f}, Difference: {diff:.4f}'
        )

    print()
    print(f'Average TEDS score: {average_score_teds}')
    print(f'Average TEDS structure score: {average_scores_teds_structure}')


if __name__ == '__main__':
    main()
