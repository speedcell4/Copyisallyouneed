import numpy as np
from tqdm import tqdm
import argparse
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from bart_score import BARTScorer


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default='gpt2_result.json')
    parser.add_argument("--device", type=int)
    return parser.parse_args()


def load_result(path):
    with open(path) as f:
        test_set = json.load(f)
        dataset = []
        for item in tqdm(test_set):
            prefix = item['prefix']
            reference = item['reference']
            result = item['text']
            reference_ids = vocab.encode(reference, add_special_tokens=False)
            if len(reference_ids) > 0:
                dataset.append((reference, prefix + ' ' + result))
    print(f'[!] collect {len(dataset)} samples')
    return dataset


if __name__ == "__main__":
    args = vars(parse_config())
    batch_size = 4
    vocab = AutoTokenizer.from_pretrained('gpt2')
    dataset = load_result(args["test_path"])
    bart_scorer = BARTScorer(device=f'cuda:{args["device"]}', checkpoint='facebook/bart-large')
    with torch.no_grad():
        scores = []
        for i in tqdm(range(len(dataset))):
            reference, result = dataset[i]
            s = bart_scorer.score([result], [reference], batch_size=4)
            scores.append(s)
        s = round(np.mean(s), 4)
    print('Results for', args['test_path'], 'BARTScore:', s, 'Dataset size', len(dataset),
          file=open(f'{args["test_path"]}_bartscore_result.txt', 'w'))
