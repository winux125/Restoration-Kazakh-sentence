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