import argparse
import json

import numpy as np
from evaluate import load
from tqdm import tqdm


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default='gpt2_result.json')
    return parser.parse_args()


def load_result(path):
    with open(path) as f:
        test_set = json.load(f)
        dataset = []
        for item in tqdm(test_set):
            prefix = item['prefix']
            reference = item['reference']
            result = item['text']

            reference = prefix + ' ' + reference
            result = prefix + ' ' + result
            dataset.append((reference, result))
    print(f'[!] collect {len(dataset)} samples')
    return dataset


if __name__ == "__main__":
    args = vars(parse_config())
    dataset = load_result(args["test_path"])

    bleurt = load('bleurt', module_type='metric', checkpoint='bleurt-large-512')
    scores = []
    for reference, result in tqdm(dataset):
        result = bleurt.compute(references=[reference], predictions=[result])
        scores.append(result['scores'][0])
    print('Results for', args['test_path'], 'BLEURT:', round(np.mean(scores), 4), 'Dataset size', len(dataset),
          file=open(f'{args["test_path"]}_bleurt_result.txt', 'w'))
