# -*- coding: utf-8 -*-
from jsonargparse import CLI
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import re

import spacy


spacy.prefer_gpu()


def is_masc(token):
    gender = token.morph.get("Gender")
    return gender and gender[0] == "Masc" 

def keep_masc(a,b):
    if is_masc(a):
        return a
    return b


# expressions for preproc
extra_spaces = re.compile(r'\s+')
SEP ="·"
sep_sub = rf"\1{SEP}\2"
extra_seps = re.compile(r'(\w)[\.•](\w)')

# expressions for sub
fem_suffix = r"(esse|sse|e|euse|se|ienne|enne|nne|ne|ère|ere|re|trice|rice|ice)"
inclusif_before_sub = r"\2\3"
inclusif_before_expr = re.compile(rf"{fem_suffix}s?{SEP}(\w+)(s?)\b")
inclusif_after_sub = r"\1\3"
inclusif_after_expr = re.compile(rf"{SEP}?(\w+)?{SEP}{fem_suffix}(s?)\b")


def preproc(text):
    # common preproc: strip extra spaces
    text = extra_spaces.sub(' ', text.strip())
    # keep a single separator : "·"
    text = extra_seps.sub(sep_sub, text)
    # hack : run twice in case of things like "tou.s.tes" where "u.s" matching will prevent "s.t" matching
    text = extra_seps.sub(sep_sub, text)
    
    return text


def sub(text):
    text = inclusif_before_expr.sub(inclusif_before_sub,text)
    text = inclusif_after_expr.sub(inclusif_after_sub,text)
    return text


def exclure(tokens):
    for sentence in tokens.sents:
        todo = []
        for token in sentence:
            if token.dep_=="ROOT" or token.pos_ in {"VERB","AUX"} or token.is_space:
                continue
            if token.lemma == token.head.lemma and token.text!=token.head.text:
                print(token,token.head,token.dep_,token.pos_,token.head.dep_,token.head.pos_,end=" ; ")   
                i = token.i-sentence.start
                j = token.head.i-sentence.start
                if j < i:
                    i, j = j, i
                todo.append((i, j, keep_masc(token.head,token)))
        if not todo:
            x_sentence = sub(sentence.text)
            if x_sentence != sentence.text:                
                training_pair = (sentence.text, x_sentence)
                yield training_pair   
            continue
        pi, pj, ptoken = todo[0]
        x_sentence = [sentence[:pi].text_with_ws, ptoken.text_with_ws]
        for i, j, token in todo[1:]:
            x_sentence.append(sentence[pj+1:i].text_with_ws)
            x_sentence.append(token.text_with_ws)
            pi, pj, ptoken = i, j, token
        x_sentence.append(sentence[pj+1:].text_with_ws)
        x_sentence = "".join(x_sentence)
        x_sentence = sub(x_sentence)
        training_pair = (sentence.text, x_sentence)
        yield training_pair       
        
        
def exclure_batch(texts, model):
    texts = [preproc(text) for text in texts]
    training_pairs = []
    for tokens in model.pipe(texts, batch_size=256):
        for training_pair in exclure(tokens):
            training_pairs.append("\t".join(training_pair))
        
    
def main(input_path_root: Path, output_path_root: Path):
    output_path_root.mkdir(exist_ok=True)
    model = spacy.load("fr_dep_news_trf", disable="ner")
    for input_path in tqdm(list(input_path_root.glob('*.jsonl'))):
        output_path = output_path_root/(input_path.with_suffix(".tsv")).name
        if output_path.exists():
            continue
        texts = pd.read_json(input_path, lines=True).content
        training_pairs = exclure_batch(texts, model)
        with open(output_path, 'wt') as file:
            file.write("\n".join(training_pairs))
        
        

if __name__ == "__main__":
    CLI(main)