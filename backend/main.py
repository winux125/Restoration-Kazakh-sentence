from fastapi import FastAPI
from pathlib import Path
from .inference import BestCRFClassifier
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchcrf import CRF

LABEL2ID = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LEN  = 512
device   = 'cuda' if torch.cuda.is_available() else 'cpu'


base_path = Path(__file__).resolve().parent
model_path = base_path.parent / 'model' / 'best_trained' / 'model.pt'

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
model     = BestCRFClassifier('xlm-roberta-large', num_labels=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()
print('Model ready!')


QUESTION_PARTICLES = {'ма','ме','ба','бе','па','пе','ше','ші'}
@torch.no_grad()
def predict(text: str) -> str:
    words = text.lower().split()
    if not words:
        return text

    enc      = tokenizer(words, is_split_into_words=True,
                         max_length=MAX_LEN, truncation=True,
                         return_tensors='pt').to(device)
    logits   = model(**enc).logits
    avg      = logits[0].unsqueeze(0)
    mask     = torch.ones(1, avg.shape[1], dtype=torch.bool).to(device)
    decoded  = model.crf.decode(avg, mask=mask)[0]


    word_ids   = enc.word_ids()
    word_preds = {}
    for i, wid in enumerate(word_ids):
        if wid is not None and wid not in word_preds:
            word_preds[wid] = ID2LABEL[decoded[i]]

    labels = [word_preds.get(i, 'O') for i in range(len(words))]


    for i, word in enumerate(words):
        if word in QUESTION_PARTICLES:
            labels[i] = 'QUESTION'

    result = []
    for word, label in zip(words, labels):
        if label == 'COMMA':
            result.append(word + ',')
        elif label == 'PERIOD':
            result.append(word + '.')
        elif label == 'QUESTION':
            result.append(word + '?')
        else:
            result.append(word)
    sentence = ' '.join(result)
    return sentence[0].upper() + sentence[1:] + (' '.join(labels)) if sentence else sentence
app = FastAPI()
@app.get('/')
async def root():
    return {"message":"ok"}

@app.post('/restore')
async def restore(text: str):
    return {"message": predict(text)}