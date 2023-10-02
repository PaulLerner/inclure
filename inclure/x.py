# -*- coding: utf-8 -*-
from tqdm import tqdm

import pandas as pd
import re

import spacy
from spacy.lang.fr import French

from .common import Path, CLI, SEP, preproc, LETTER


spacy.prefer_gpu()


def is_masc(token):
    gender = token.morph.get("Gender")
    return gender and gender[0] == "Masc" 

def different_gender(a, b):
    ga = a.morph.get("Gender")
    gb = b.morph.get("Gender")
    return (ga and gb) and (ga != gb)

def keep_masc(a,b):
    if is_masc(a):
        return a
    return b


fem_suffix = r"(esse|sse|e|euse|se|ienne|enne|nne|ne|ère|ere|re|trice|rice|ice)"
# common extensions not to be confused with feminine suffixes
domain_and_exts = re.compile(rf'\w+({SEP}(com|fr|net|org|html|be|ca|info|ch|style|screenshots|free|jpg|js|php|google|canadiantire))\b')
pre_sub = r"\1\2"
pre_expr = re.compile(rf"({LETTER}+){SEP}({LETTER}+{SEP}{LETTER}+)")
inclusif_before_sub = r"\2"
inclusif_before_expr = re.compile(rf"{fem_suffix}s?{SEP}({LETTER}+)\b")
inclusif_after_sub = r"\1\3"
inclusif_after_expr = re.compile(rf"({LETTER}+){SEP}{fem_suffix}(s?)\b")
fix_sep = re.compile(SEP)
fix_ss = re.compile(r"ss\b")


def sub(text):
    # remove extra seps e.g. "au.trices.teurs" -> "autrices.teurs"
    text = pre_expr.sub(pre_sub, text)
    before = text
    text = inclusif_before_expr.sub(inclusif_before_sub, text)
    text = inclusif_after_expr.sub(inclusif_after_sub, text)
    if text == before:
        return None
    # fixes erratic uses e.g. "tous.tes" vs. "tou.te.s"
    text = fix_sep.sub("", text)
    text = fix_ss.sub("s", text)
    return text


def exclure(tokens):
    for sentence in tokens.sents:
        if domain_and_exts.search(sentence.text) is not None:
            continue
        todo = []
        bad_sentence = False
        for token in sentence:
            if token.like_url or token.like_email:
                bad_sentence = True
                break
            if token.dep_=="ROOT" or token.pos_ in {"VERB","AUX"} or token.is_space:
                continue
            if token.lemma == token.head.lemma and different_gender(token, token.head):
                i = token.i-sentence.start
                j = token.head.i-sentence.start
                if j < i:
                    i, j = j, i
                todo.append((i, j, keep_masc(token.head,token)))
        if bad_sentence:
            continue
        if not todo:
            x_sentence = sub(sentence.text)
            if x_sentence is not None and x_sentence != sentence.text:                
                training_pair = (sentence.text.strip(), x_sentence.strip())
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
        xx_sentence = sub(x_sentence)
        if xx_sentence is None:
            xx_sentence = x_sentence
        training_pair = (sentence.text.strip(), xx_sentence.strip())
        yield training_pair       
        
        
def exclure_batch(texts, sentencizer, model):
    texts = [preproc(text) for text in texts]
    sents = []
    for tokens in tqdm(sentencizer.pipe(texts), desc="sentencizing"):
        for sent in tokens.sents:            
            sents.append(sent.text)
    training_pairs = []
    for tokens in tqdm(model.pipe(sents, batch_size=2048), desc="excluding"):
        for training_pair in exclure(tokens):
            training_pairs.append("\t".join(training_pair))
    return training_pairs
        
    
def main(input_path_root: Path, output_path_root: Path):
    output_path_root.mkdir(exist_ok=True)
    sentencizer = French()
    sentencizer.max_length = int(1e12)
    sentencizer.add_pipe('sentencizer')
    model = spacy.load("fr_dep_news_trf", disable="ner")
    for input_path in input_path_root.glob('*.jsonl'):
        print(input_path)
        output_path = output_path_root/(input_path.with_suffix(".tsv")).name
        if output_path.exists():
            continue
        texts = pd.read_json(input_path, lines=True).content
        training_pairs = exclure_batch(texts, sentencizer, model)
        with open(output_path, 'wt') as file:
            file.write("\n".join(training_pairs))
        
        
if __name__ == "__main__":
    CLI(main)