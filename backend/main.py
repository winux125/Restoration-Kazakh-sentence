from fastapi import FastAPI
from pathlib import Path
from .inference import BestCRFClassifier
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchcrf import CRF
from contextlib import asynccontextmanager
from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

LABEL2ID = {'O': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LEN  = 512
device   = 'cuda' if torch.cuda.is_available() else 'cpu'


base_path = Path(__file__).resolve().parent
model_path = base_path.parent / 'model' / 'best_trained' / 'model.pt'


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Загружается один раз при старте
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    model = BestCRFClassifier('xlm-roberta-large', num_labels=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print('Model ready!')
    yield
    # При остановке сервера

app = FastAPI(lifespan=lifespan)


QUESTION_PARTICLES = {'ма','ме','ба','бе','па','пе','ше','ші'}
@torch.no_grad()
def predict(text: str) -> str:
    words = text.lower().split()
    if not words:
        return text

    n         = len(words)
    win_size  = MAX_LEN // 2
    logit_sum = torch.zeros(n, len(LABEL2ID))
    logit_cnt = torch.zeros(n)

    starts = list(range(0, n, 100))
    if not starts or starts[-1] + win_size < n:
        starts.append(max(0, n - win_size))

    for start in starts:
        chunk = words[start:start + win_size]
        enc   = tokenizer(chunk, is_split_into_words=True,
                          max_length=MAX_LEN, truncation=True,
                          return_tensors='pt').to(device)
        logits   = model(**enc).logits[0].cpu().float()
        word_ids = enc.word_ids()
        seen = set()
        for tok_i, wid in enumerate(word_ids):
            if wid is None or wid in seen: continue
            gid = start + wid
            if gid < n:
                logit_sum[gid] += logits[tok_i]
                logit_cnt[gid] += 1
            seen.add(wid)

    avg     = (logit_sum / logit_cnt.clamp(min=1).unsqueeze(-1)).unsqueeze(0).to(device)
    mask    = torch.ones(1, n, dtype=torch.bool).to(device)
    decoded = model.crf.decode(avg, mask=mask)[0]

    labels = [ID2LABEL[decoded[i]] for i in range(n)]
    for i, word in enumerate(words):
        if word in QUESTION_PARTICLES:
            labels[i] = 'QUESTION'

    result = []
    for word, label in zip(words, labels):
        if label == 'COMMA':   result.append(word + ',')
        elif label == 'PERIOD': result.append(word + '.')
        elif label == 'QUESTION': result.append(word + '?')
        else: result.append(word)

    sentence = ' '.join(result)
    return sentence[0].upper() + sentence[1:] if sentence else sentence


@app.get('/')
async def root():
    return {"message":"ok"}

@app.post('/restore')
async def restore(request: TextRequest):
    try:
        result = predict(request.text)
        return {'result': result}
    except Exception as e:
        return {'error': str(e)}, 500