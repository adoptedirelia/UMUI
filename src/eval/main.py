"""
Evaluation utilities for claim verification models.
Supports accuracy, F1, MSE, NLL, ECE, and Krippendorff's alpha.
"""

import json
from matplotlib import pyplot as plt
import os
import re
from src.analyse.analyse import calculate_krippendorff_alpha, calculate_binned_data
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

THRESHOLD = 0.5


def extract_answer_from_output(output_text: str) -> float:
    if 'assistant' not in output_text:
        pattern = r'<answer>([^<]+)</answer>'
        matches = re.findall(pattern, output_text, re.IGNORECASE)

        if matches:
            answer_str = matches[-1].strip()
            try:
                value = float(answer_str)
                if value > 1.0:
                    value = value / 100.0
                return max(0.0, min(1.0, value))
            except (ValueError, TypeError):
                return 0.0
        return 0.0
    else:
        try:
            text = 'assistant\n<answer>'
            result = output_text.split(text)[-1].split('<')[0]
            return float(result)
        except:
            return 0.0


def extract_answer_from_output_list(output_text: list, method: str = 'mean') -> float:
    lst = [extract_answer_from_output(item) for item in output_text]
    if method == 'mean':
        return np.mean(lst)
    else:
        return max(lst)


def load_data(folder: str, dataname: str) -> list:
    unfold_data = []
    for file in os.listdir(folder):
        if file.endswith('_0_824.json'):
            continue
        if not file.startswith(dataname):
            continue
        if file.endswith('.json'):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            unfold_data.extend([item for sublist in data.values() for item in sublist])
    return unfold_data


def filter_data(data: list, unfold_data: list, eval_file: str) -> list:
    """Leave eval data only. Requires a reference eval file path."""
    with open(eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    unfold_data2 = [item for sublist in data.values() for item in sublist]
    lst = []
    for item in unfold_data:
        for item2 in unfold_data2:
            if (item['path'], item['claim']) == (item2['path'], item2['claim']):
                lst.append(item)
                break
    return lst


def result(data: list) -> tuple:
    pred_list = []
    label_list = []
    prob_list = []
    for item in data:
        pred_list.append(1 if extract_answer_from_output(item['answer']) > THRESHOLD else 0)
        label_list.append(item['label'])
        prob_list.append(extract_answer_from_output(item['answer']))
    return pred_list, label_list, prob_list


def human_analysis(human_results: str, data2: list) -> tuple:
    with open(human_results, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    if isinstance(data1, dict):
        data1 = [item for sublist in data1.values() for item in sublist]

    data2_new = []
    for item in data1:
        for item2 in data2:
            if item['claim'] == item2['claim'] and item['path'].split('/')[-1] == item2['path'].split('/')[-1]:
                data2_new.append(item2)
                break

    probability1 = []
    probability2 = []
    label_list1 = []
    label_list2 = []
    pre_list1 = []
    pre_list2 = []
    for i in data1:
        probability1.append(i['probability'])
        pre_list1.append(1 if i['probability'] > THRESHOLD else 0)
        path = i['path'].split('/')[-1]
        claim = i['claim']
        for item in data2_new:
            if item['path'].split('/')[-1] == path and item['claim'] == claim:
                probability2.append(extract_answer_from_output(item['answer']))
                label_list2.append(item['label'])
                label_list1.append(item['label'])
                pre_list2.append(1 if extract_answer_from_output(item['answer']) > THRESHOLD else 0)
                break

    binned_data = calculate_binned_data([probability1, probability2], bin_num=10)
    alpha = calculate_krippendorff_alpha(binned_data)
    print(alpha)
    mse = np.mean((np.array(probability1) - np.array(probability2)) ** 2)
    print(mse)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.hist(probability1, bins=50, edgecolor='black', label=f'human,acc={accuracy_score(label_list1, pre_list1):.4f},f1={f1_score(label_list1, pre_list1):.4f}')
    plt.title(f"Result Distribution(human)")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(probability2, bins=50, edgecolor='black', label=f'model,acc={accuracy_score(label_list2, pre_list2):.4f},f1={f1_score(label_list2, pre_list2):.4f}')
    plt.title(f"Result Distribution(model)")
    plt.legend()
    plt.savefig(f'human_model_distribution.png')


def quick_eval(data_path: str) -> tuple:
    pred_list = []
    label_list = []
    prob_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    unfold_data = [item for sublist in data.values() for item in sublist]
    if 'unli' not in data_path:
        new_lst = []
        set_ = set()
        for item in unfold_data:
            if (item['claim'], item['path']) in set_:
                continue
            set_.add((item['claim'], item['path']))
            new_lst.append(item)
        unfold_data = new_lst

    for item in unfold_data:
        pred_list.append(1 if extract_answer_from_output(item['answer']) > THRESHOLD else 0)
        label_list.append(item['label'])
        prob_list.append(extract_answer_from_output(item['answer']))
    print(f"acc={accuracy_score(label_list, pred_list):.4f},f1={f1_score(label_list, pred_list):.4f}")
    print('true: ', label_list.count(1), 'false: ', label_list.count(0))
    plt.figure(figsize=(15, 5))
    plt.hist(prob_list, bins=50, edgecolor='black', label=f'model,acc={accuracy_score(label_list, pred_list):.4f},f1={f1_score(label_list, pred_list):.4f}')
    plt.title(f"Result Distribution(model)")
    plt.legend()
    plt.savefig(f'model_distribution_qe.png')


def quick_eval_human(data_path: str, modality: str) -> tuple:
    pred_list = []
    label_list = []
    prob_list = []
    prob_list2 = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    unfold_data = [item for sublist in data.values() for item in sublist]

    for item in unfold_data:
        pred_list.append(1 if extract_answer_from_output(item['answer']) > THRESHOLD else 0)
        if modality == 'audio':
            label_list.append(item['label']['audio'])
        elif modality == 'video':
            label_list.append(item['label']['video'])
        elif modality == 'omni':
            label_list.append(item['label']['omni'])

        prob_list.append(extract_answer_from_output(item['answer']))
        prob_list2.append(item['probability'])

    print(f"acc={accuracy_score(label_list, pred_list):.4f},f1={f1_score(label_list, pred_list):.4f}")
    print('true: ', label_list.count(1), 'false: ', label_list.count(0))
    binned_data = calculate_binned_data([prob_list, prob_list2], bin_num=10)
    alpha = calculate_krippendorff_alpha(binned_data)
    print(alpha)
    mse = np.mean((np.array(prob_list) - np.array(prob_list2)) ** 2)
    print(mse)


def quick_eval_model(data_path: str, modality: str) -> tuple:
    pred_list = []
    label_list = []
    prob_list = []
    prob_list2 = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    unfold_data = data
    new_lst = []
    set_ = set()
    for item in unfold_data:
        if (item['claim'], item['path']) in set_:
            continue
        set_.add((item['claim'], item['path']))
        new_lst.append(item)
    unfold_data = new_lst
    for item in unfold_data:
        probability = extract_answer_from_output_list(item['output'])
        pred_list.append(1 if probability > THRESHOLD else 0)
        if modality == 'audio':
            label_list.append(item['label']['audio'])
        elif modality == 'video':
            label_list.append(item['label']['video'])
        elif modality == 'omni':
            label_list.append(item['label']['omni'] or item['label']['audio'] or item['label']['video'])

        prob_list.append(probability)
        prob_list2.append(item['probability'])

    print(f"acc={accuracy_score(label_list, pred_list):.4f},f1={f1_score(label_list, pred_list):.4f}")
    print('true: ', label_list.count(1), 'false: ', label_list.count(0))
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.hist(prob_list, bins=50, edgecolor='black')
    plt.title(f"Result Distribution(model)")
    plt.subplot(1, 2, 2)
    plt.hist(prob_list2, bins=50, edgecolor='black')
    plt.title(f"Result Distribution(human)")
    plt.legend()
    plt.savefig(f'model_human_distribution.png')

    binned_data = calculate_binned_data([prob_list, prob_list2], bin_num=10)
    alpha = calculate_krippendorff_alpha(binned_data)
    print(alpha)
    mse = np.mean((np.array(prob_list) - np.array(prob_list2)) ** 2)
    print(mse)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def nll(y, p):
    p = np.clip(np.array(p), 1e-10, 1 - 1e-10)
    return -np.mean(np.array(y) * np.log(p) + (1 - np.array(y)) * np.log(1 - p))


def ece(y, p, bins=10):
    y = np.array(y)
    p = np.array(p)
    bin_edges = np.linspace(0, 1, bins + 1)
    ece_val = 0
    N = len(y)
    for i in range(bins):
        mask = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
        if np.sum(mask) > 0:
            conf = np.mean(p[mask])
            acc = np.mean(y[mask])
            ece_val += np.sum(mask) / N * abs(acc - conf)
    return ece_val


def eval_mse(data_path: str, unli_validation_path: str = ''):
    lst = []
    with open(unli_validation_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            lst.append(data)
    with open(data_path, 'r') as f:
        data = json.load(f)

    label_prob_list = []
    pred_prob_list = []

    def find_prob_in_unli(claim):
        for item in lst:
            if item['hypothesis'] == claim:
                return item['label']
        return None

    unfold_data = [item for sublist in data.values() for item in sublist]
    for item in unfold_data:
        label_prob_list.append(find_prob_in_unli(item['claim']))
        pred_prob_list.append(extract_answer_from_output(item['answer']))
    mse = np.mean((np.array(label_prob_list) - np.array(pred_prob_list)) ** 2)
    nll_score = nll(label_prob_list, pred_prob_list)
    ece_score = ece(label_prob_list, pred_prob_list)
    print('mse: ', mse, 'nll_score: ', nll_score, 'ece_score: ', ece_score)


def eval_mse_wikivideo(data_path: str, modality: str, reference_file: str = ''):
    with open(data_path, 'r') as f:
        data = json.load(f)
    unfold_data = [item for sublist in data.values() for item in sublist]
    with open(reference_file, 'r') as f:
        data2 = json.load(f)
    unfold_data2 = data2

    label_prob_list = []
    pred_prob_list = []
    label_list = []
    pred_list = []
    for item in unfold_data:
        claim = item['claim']
        v = item['path'].split('/')[-1].split('.')[0]
        for item2 in unfold_data2:
            if item2['claim'] == claim and item2['path'].split('/')[-1].split('.')[0] == v:
                label_prob_list.append(item2['probability'])
                pred_prob_list.append(extract_answer_from_output(item['answer']))
                pred_list.append(1 if extract_answer_from_output(item['answer']) > THRESHOLD else 0)
                if modality == 'audio':
                    label_list.append(item2['label']['audio'])
                elif modality == 'video':
                    label_list.append(item2['label']['video'])
                elif modality == 'omni':
                    label_list.append(item2['label']['omni'] or item2['label']['audio'] or item2['label']['video'])
                break
    mse = np.mean((np.array(label_prob_list) - np.array(pred_prob_list)) ** 2)
    nll_score = nll(label_prob_list, pred_prob_list)
    ece_score = ece(label_prob_list, pred_prob_list)

    print('mse: ', mse, 'nll_score: ', nll_score, 'ece_score: ', ece_score)
    print(f"acc={accuracy_score(label_list, pred_list):.4f},f1={f1_score(label_list, pred_list):.4f}")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.hist(pred_prob_list, bins=50, edgecolor='black')
    plt.title(f"Result Distribution(model)")
    plt.subplot(1, 2, 2)
    plt.hist(label_prob_list, bins=50, edgecolor='black')
    plt.title(f"Result Distribution(human)")
    plt.legend()
    plt.savefig(f'model_human_distribution_wikivideo.png')
    return label_prob_list, pred_prob_list


def training_data_eval(omni_path: str = ''):
    with open(omni_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = [item for sublist in data.values() for item in sublist]

    label_list = []
    pre_list = []
    for item in data:
        label_list.append(item['label'])
        pre_list.append(1 if extract_answer_from_output_list(item['answer'], method='mean') > 0.5 else 0)

    print(accuracy_score(label_list, pre_list))


def quick_eval_binary(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    unfold_data = [item for sublist in data.values() for item in sublist]
    label_list = []
    pre_list = []
    for item in unfold_data:
        if 'no' in item['answer'] or 'No' in item['answer']:
            pre_list.append(0)
        else:
            pre_list.append(1)
        label_list.append(item['label'])
    print(accuracy_score(label_list, pre_list))
    print(f1_score(label_list, pre_list))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', type=str, default='mse', choices=['mse', 'wikivideo', 'binary', 'quick'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--unli_validation_path', type=str, default='')
    parser.add_argument('--reference_file', type=str, default='')
    parser.add_argument('--modality', type=str, default='video')
    args = parser.parse_args()

    if args.eval_type == 'mse':
        eval_mse(args.data_path, unli_validation_path=args.unli_validation_path)
    elif args.eval_type == 'wikivideo':
        eval_mse_wikivideo(args.data_path, args.modality, reference_file=args.reference_file)
    elif args.eval_type == 'binary':
        quick_eval_binary(args.data_path)
    elif args.eval_type == 'quick':
        quick_eval(args.data_path)
