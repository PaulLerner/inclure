# -*- coding: utf-8 -*-
import json

from datasets import Dataset

from .common import CLI, Path


def main(data_root_path: Path):
    """Split data in train-dev-test"""
    data = []
    for path in data_root_path.glob('*.json'):
        with open(path, 'rt') as file:
            subset = json.load(file)
        data.extend(subset)
    dataset = Dataset.from_dict({'translation': data})
    print(dataset)
    dataset = dataset.train_test_split(test_size=0.2, seed=0)
    dev_test = dataset['test'].train_test_split(test_size=0.5, seed=0)
    dataset.pop('test')
    dataset['validation'] = dev_test['train']
    dataset['test'] = dev_test['test']
    print(dataset)
    dataset.save_to_disk(data_root_path)    
     
        
if __name__ == "__main__":
    CLI(main, description=main.__doc__)
