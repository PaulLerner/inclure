# -*- coding: utf-8 -*-
"""
usage: data.py [-h] [--config CONFIG] [--print_config [={comments,skip_null,skip_default}+]]
               data_root_path

options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config [={comments,skip_null,skip_default}+]
                        Print the configuration after applying all other arguments and exit.

<function main at 0x7f6c10018ee0>:
  data_root_path        (required, type: <class 'Path'>)
"""

from datasets import Dataset

from .common import CLI, Path


def main(data_root_path: Path):
    data = []
    for path in data_root_path.glob('*.tsv'):
        with open(path, 'rt') as file:
            for line in file.readlines():
                if not line:
                    continue
                pair = line.split("\t")
                if len(pair) != 2:
                    breakpoint()
                data.append({'fri': pair[0], 'fr': pair[1]})
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
    CLI(main)