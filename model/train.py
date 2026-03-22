import os, re, random, pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.optimize import minimize
from scipy.special import softmax

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer, AutoModel,
    TrainingArguments, Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import TokenClassifierOutput
from torchcrf import CRF
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

class CFG:
    #--MODEL---------------------------------------------------------------
    MODEL_NAME = "xlm-roberta-large"

    #--LABELS--------------------------------------------------------------
    LABEL2ID = {'O': 0,'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3}
    ID2LABEL = {v: k for k,v in LABEL2ID.items()}
    IGNORE_INDEX = -100

    #--HYPERPARAMETRS------------------------------------------------------
    MAX_LEN = 512
    BATCH_SIZE = 32
    GRAD_ACCUM = 2
    LR = 1e-5
    EPOCHS = 5

    #--DATA----------------------------------------------------------------
    N_MULTIDOMAIN = 200000 #text from the multidomain dataset
    N_WIKI = 50000 #text from the wikipedia KK dataset
    MAX_TRAIN = 120000 #max amount of example for training
    
    #--saved model---------------------------------------------------------
    CHECKPOINT_PT = './best_trained/model.pt'
    CHECKPOINT_DIR = './best_trained'

    #--output data---------------------------------------------------------
    SAVE_DIR = './trained' #where to save the model
    DATA_DIR  = Path('data');  DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR = Path('model'); MODEL_DIR.mkdir(exist_ok=True) 

    #--competition files---------------------------------------------------
    def __init__(self):
        self.test_df       = pd.read_csv('./data/test.csv')
        self.train_example = pd.read_csv('./data/train_example.csv')

    def competition_files(self):
        print(f'Train example: {len(self.train_example)} rows')
        print(f'Test:          {len(self.test_df)} rows')



class PreProcessing:
    KAZ_PUNCT = re.compile(r'[«»"\(\)\[\]\{\};:\-–—/\\]')
    URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+')

    def strip_and_label(self, text: str):
        """Removes Punctuation from the text and notes where it stood as labels"""
        text = self.URL_PATTERN.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()  

        tokens = text.split()
        clean, labels = [], []
        for tok in tokens:
            tok = self.KAZ_PUNCT.sub('', tok).strip()

            if not tok:
                continue
            if tok.endswith('?'):
                label, tok = 'QUESTION', tok[:-1]
            elif tok.endswith(('!','.','…')):
                label, tok = 'PERIOD', tok.rstrip('!.…')
            elif tok.endswith(','):
                label, tok = 'COMMA', tok[:-1]
            else:
                label = 'O'
            tok = tok.lower().strip()
            if tok:
                clean.append(tok)
                labels.append(label)

        return clean, labels
    def texts_to_rows(self, texts, max_words=50, min_words=8):
        """Converts texts to strings of dataests(input_text, labels)"""
        rows = []
        sent_splitter = re.compile(r'(?<=[.!?])\s+')
        for text in texts:
            if not isinstance(text, str) or len(text) < 20:
                continue
            buf_tok, buf_lbl = [], []
            for sent in sent_splitter.split(text.strip()):
                tokens, labels = self.strip_and_label(sent)
                if not tokens:
                    continue
                if len(buf_tok) + len(tokens) > max_words and len(buf_tok) >= min_words:
                    rows.append({'input_text': ' '.join(buf_tok),
                                'labels':     ' '.join(buf_lbl)})
                    buf_tok, buf_lbl = [], []
                buf_tok.extend(tokens)
                buf_lbl.extend(labels)
            if len(buf_tok) >= min_words:
                rows.append({'input_text': ' '.join(buf_tok),
                            'labels':     ' '.join(buf_lbl)})
        return rows


        
    def is_quality_row(self,labels_str):
        """Фильтр качества: строка должна содержать COMMA и быть достаточно пунктуированной."""
        labels = labels_str.split()
        if 'COMMA' not in labels:
            return False
        return labels.count('O') / len(labels) < 0.95

class TrainingData:
    def __init__(self):
        self.all_rows = []
        self.preprocessor = PreProcessing()  # ← добавить

    def stream_dataset(self,hf_path, n_target, text_col='text',
                   batch_size=10_000, split_paragraphs=False, **load_kwargs):
        ds = load_dataset(hf_path, split='train', streaming=True, **load_kwargs)
        ds = ds.shuffle(seed=SEED, buffer_size=50_000)
        rows, batch, collected = [], [], 0
        for row in ds:
            text = row.get(text_col, '')
            if 'predicted_language' in row and row['predicted_language'] != 'kaz':
                continue
            if 'contains_kaz_symbols' in row and row['contains_kaz_symbols'] != 1:
                continue
            if not isinstance(text, str) or len(text) < 30:
                continue
            if split_paragraphs:
                batch.extend([p.strip() for p in text.split('\n') if len(p.strip()) > 30])
            else:
                batch.append(text)
            if len(batch) >= batch_size:
                rows.extend(self.preprocessor.texts_to_rows(batch))
                collected += len(batch)
                batch = []
                print(f'  {collected:,} texts -> {len(rows):,} rows')
            if collected >= n_target:
                break
        if batch:
            rows.extend(self.preprocessor.texts_to_rows(batch))
        return rows
    def sourceRead(self,cfg):
        self.all_rows.extend(self.stream_dataset(
            'kz-transformers/multidomain-kazakh-dataset', n_target=cfg.N_MULTIDOMAIN
        ))
        print(f'After source 1: {len(self.all_rows):,}')

        try:
            self.all_rows.extend(self.stream_dataset(
                'wikimedia/wikipedia', n_target=cfg.N_WIKI,
                text_col='text', split_paragraphs=True, name='20231101.kk',
            ))
            print(f'After source 2: {len(self.all_rows):,}')
        except Exception as e:
            print(f'Wikipedia failed: {e}')
        for _,r in cfg.train_example.iterrows():
            self.all_rows.append({"input_text": r['input_text'], 'labels': r['labels']})
        print(f'Total: {len(self.all_rows)}')


def quality_filter(all_rows):
    return all_rows