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

BASE_PATH = Path(__file__).resolve().parent
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
    CHECKPOINT_PT = BASE_PATH / 'best_trained' / 'model.pt'
    CHECKPOINT_DIR = BASE_PATH / 'best_trained'

    #--output data---------------------------------------------------------
    SAVE_DIR  = BASE_PATH / 'trained' #where to save the model
    DATA_DIR  = BASE_PATH / 'data';  DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR = BASE_PATH / 'model'; MODEL_DIR.mkdir(exist_ok=True) 

    #--competition files---------------------------------------------------
    def __init__(self):
        self.test_df       = pd.read_csv(BASE_PATH / 'data' / 'test.csv')
        self.train_example = pd.read_csv(BASE_PATH / 'data' / 'train_example.csv')
        self.SAVE_DIR.mkdir(exist_ok=True)       
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)

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


def quality_filter(all_rows,cfg):
    df = pd.DataFrame(all_rows).drop_duplicates('input_text').reset_index(drop=True)
    print(f'Before: {len(df)}')

    pp = PreProcessing()
    df_quality = df[df['labels'].apply(pp.is_quality_row)].reset_index(drop=True)
    df_other   = df[~df['labels'].apply(pp.is_quality_row)]
    n_other    = min(len(df_quality) // 2,len(df_other))
    df = pd.concat([df_quality, df_other.sample(n_other, random_state=SEED)])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f'After filter: {len(df):,}')
    
    if len(df) > cfg.MAX_TRAIN + 5000:
        df = df[:cfg.MAX_TRAIN + 5000]
    all_labels = ' '.join(df['labels']).split()
    cnt = Counter(all_labels)
    total = sum(cnt.values())
    print('Label distribution:')
    for lbl, c in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f'  {lbl:10s}: {c:>10,}  ({100*c/total:.1f}%)')
    val_size = min(5000, int(len(df) * 0.05))
    val_df   = df[:val_size]
    train_df = df[val_size:]
    print(f'Train: {len(train_df):,}  |  Val: {len(val_df):,}')
    return val_size,val_df,train_df


def encode_row(words, labels, tokenizer, cfg):
    enc = tokenizer(
        words, is_split_into_words=True,
        max_length=cfg.MAX_LEN, truncation=True, padding=False,
    )
    word_ids, label_ids, prev_word = enc.word_ids(), [], None
    for wid in word_ids:
        if wid is None:
            label_ids.append(cfg.IGNORE_INDEX)
        elif wid != prev_word:
            label_ids.append(cfg.LABEL2ID[labels[wid]] if wid < len(labels) else cfg.IGNORE_INDEX)
        else:
            label_ids.append(cfg.IGNORE_INDEX)
        prev_word = wid
    enc['labels'] = label_ids
    return enc


class PunctDataset(Dataset):
    def __init__(self, df, tokenizer, cfg):
        self.samples = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc='Encoding'):
            words  = row['input_text'].split()
            labels = row['labels'].split()
            if len(words) != len(labels):
                continue
            self.samples.append(encode_row(words, labels, tokenizer, cfg))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.samples[idx].items()}


class BertCRFClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(cfg.MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.linear  = nn.Linear(self.encoder.config.hidden_size, len(cfg.LABEL2ID))
        self.crf     = CRF(len(cfg.LABEL2ID), batch_first=True)
        self.encoder.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        logits = self.linear(self.dropout(hidden))
        loss = None
        if labels is not None:
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            crf_mask = attention_mask.bool()
            loss = -self.crf(logits, safe_labels, mask=crf_mask, reduction='mean')
        return TokenClassifierOutput(loss=loss, logits=logits)


def compute_metrics(eval_pred, cfg):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    y_true, y_pred = [], []
    for pred_row, label_row in zip(preds, labels):
        for p, l in zip(pred_row, label_row):
            if l == cfg.IGNORE_INDEX: continue
            y_true.append(cfg.ID2LABEL[l])
            y_pred.append(cfg.ID2LABEL[p])
    score = f1_score(y_true, y_pred,
                     labels=['COMMA', 'PERIOD', 'QUESTION'],
                     average='macro', zero_division=0)
    per = f1_score(y_true, y_pred,
                   labels=['COMMA', 'PERIOD', 'QUESTION'],
                   average=None, zero_division=0)
    return {
        'macro_f1':    round(score,  4),
        'f1_comma':    round(per[0], 4),
        'f1_period':   round(per[1], 4),
        'f1_question': round(per[2], 4),
    }


def train():
    cfg = CFG()

    # data
    td = TrainingData()
    td.sourceRead(cfg)

    val_size, val_df, train_df = quality_filter(td.all_rows, cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    print(f'Vocab size: {tokenizer.vocab_size:,}')

    train_dataset = PunctDataset(train_df, tokenizer, cfg)
    val_dataset   = PunctDataset(val_df,   tokenizer, cfg)
    print(f'Train: {len(train_dataset):,}  |  Val: {len(val_dataset):,}')

    # model
    model = BertCRFClassifier(cfg).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    # Training
    import shutil
    shutil.rmtree(str(cfg.MODEL_DIR), ignore_errors=True)
    cfg.MODEL_DIR.mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = str(cfg.MODEL_DIR),
        num_train_epochs            = cfg.EPOCHS,
        per_device_train_batch_size = cfg.BATCH_SIZE,
        per_device_eval_batch_size  = cfg.BATCH_SIZE * 2,
        gradient_accumulation_steps = cfg.GRAD_ACCUM,
        learning_rate               = cfg.LR,
        weight_decay                = 0.01,
        warmup_ratio                = 0.06,
        lr_scheduler_type           = 'cosine',
        eval_strategy               = 'epoch',
        save_strategy               = 'epoch',
        save_total_limit            = 1,
        load_best_model_at_end      = True,
        metric_for_best_model       = 'macro_f1',
        greater_is_better           = True,
        logging_steps               = 200,
        fp16                        = torch.cuda.is_available(),
        dataloader_num_workers      = 2,
        report_to                   = 'none',
        seed                        = SEED,
    )

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        processing_class = tokenizer,
        data_collator    = collator,
        compute_metrics  = lambda ep: compute_metrics(ep, cfg),
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    import os
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), cfg.SAVE_DIR / 'model.pt')  # ← Path объект, не str
    tokenizer.save_pretrained(str(cfg.SAVE_DIR)) 

if __name__ == '__main__':
    train()