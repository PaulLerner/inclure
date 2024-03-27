# inclure
Automatic translation from Standard to Inclusive French, and vice-versa. 

Source code and data for the paper "INCLURE: a Dataset  and Toolkit for Inclusive French Translation" (Lerner and Grouin, 2024).

## getting the INCLURE data

Using `datasets.load_dataset`, e.g.  `datasets.load_dataset("PaulLerner/oscar_inclure")`: https://huggingface.co/datasets/PaulLerner/oscar_inclure

OOD test set https://huggingface.co/datasets/PaulLerner/cfi_sents_v_taln_2022/


## experiments
### trained models
- https://huggingface.co/PaulLerner/fabien.ne_barthez
- https://huggingface.co/PaulLerner/fabien_barthez

### train your own

`python -m inclure.train experiments/inclure/train_config.json`

`python -m inclure.train experiments/exclure/train_config.json`

### evaluate

`python -m inclure.train /path/to/test_config.json`

one of `experiments/*/*/test_config.json`

## reproduce/extend INCLURE

Get OSCAR 22.01 from https://huggingface.co/datasets/oscar-corpus/OSCAR-2201

By any chance, if you have access to Jean Zay, use `$DSDIR/OSCAR/fr_meta`

`python -m inclure.x /path/to/oscar/fr_meta /output/folder`

`python -m inclure.data /output/folder`

## reference

TODO