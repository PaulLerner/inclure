# -*- coding: utf-8 -*-
from tqdm import tqdm
import json
import pandas as pd
import re

import spacy
from spacy.lang.fr import French

from .common import Path, CLI, SEP, preproc, LETTER


spacy.prefer_gpu()

def get_gender(token):
    gender = token.morph.get("Gender")
    if not gender:
        return None
    return gender[0]

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


def detect(text):
    return inclusif_before_expr.search(text) is not None or inclusif_after_expr.search(text) is not None


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


def gender_stat(tokens, gender_co):
    for sentence in tokens.sents:
        for token in sentence:
            if token.dep_=="ROOT" or token.pos_ in {"VERB","AUX"} or token.is_space:
                continue
            if token.lemma == token.head.lemma:
                i = token.i
                j = token.head.i
                if i < j:
                    gender_co[(get_gender(token), get_gender(token.head))] += 1
                else:
                    gender_co[(get_gender(token.head), get_gender(token))] += 1
           
                
def exclure(tokens):
    for sentence in tokens.sents:
        if domain_and_exts.search(sentence.text) is not None:
            continue
        todo = []
        bad_sentence = False
        for token in sentence:
            # FIXME: better detector than this: gives a lot of false positives
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
                training_pair = {"fri": sentence.text.strip(), "fr": x_sentence.strip(), "inflection": True, "coordination": False}
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
            inflection = False
            xx_sentence = x_sentence
        else:
            inflection = True
        training_pair = {"fri": sentence.text.strip(), "fr": xx_sentence.strip(), "inflection": inflection, "coordination": True}
        yield training_pair       
        
        
def exclure_batch(texts, sentencizer, model, detect_filter=False):
    proc_texts = []
    for text in tqdm(texts, desc="preprocessing"):
        text = preproc(text)
        # spacy is much more expensive than regex
        # first process the whole document to see if it contains "écriture inclusive"
        # process at the doc-level as the same doc may contain mixed-inclusive french writing styles 
        # (one detected with regex and the other later detected with spacy) so we limit false negatives
        if (not detect_filter) or (detect_filter and detect(text)):
            proc_texts.append(text)
    if detect_filter:
        print(f"Filtered {len(proc_texts)} documents out of {len(texts)}")
    sents = []
    for tokens in tqdm(sentencizer.pipe(proc_texts), desc="sentencizing"):
        for sent in tokens.sents:            
            sents.append(sent.text)
    training_pairs = []
    for tokens in tqdm(model.pipe(sents, batch_size=2048), desc="excluding"):
        for training_pair in exclure(tokens):
            training_pairs.append(training_pair)
    return training_pairs
        
    
def main(input_path_root: Path, output_path_root: Path, detect_filter: bool = False):
    """Generate inclusive to standard french sentence pairs from a large corpus (e.g. OSCAR)"""
    output_path_root.mkdir(exist_ok=True)
    sentencizer = French()
    sentencizer.max_length = int(1e12)
    sentencizer.add_pipe('sentencizer')
    model = spacy.load("fr_dep_news_trf", disable="ner")
    # to reproduce, use fr_meta_part_115.jsonl  fr_meta_part_14.jsonl  fr_meta_part_166.jsonl  fr_meta_part_283.jsonl  fr_meta_part_351.jsonl
    for input_path in input_path_root.glob('*.jsonl'):
        print(input_path)
        output_path = output_path_root/(input_path.with_suffix(".json")).name
        if output_path.exists():
            continue
        texts = pd.read_json(input_path, lines=True).content
        training_pairs = exclure_batch(texts, sentencizer, model, detect_filter=detect_filter)
        with open(output_path, 'wt') as file:
            json.dump(training_pairs, file)
        
        
if __name__ == "__main__":
    CLI(main, description=main.__doc__)
