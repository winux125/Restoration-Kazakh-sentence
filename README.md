# Restoration of Kazakh Sentence Punctuation

<p align="center">
  <img src="https://img.shields.io/badge/language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/model-XLM--RoBERTa--large-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white" alt="XLM-RoBERTa"/>
  <img src="https://img.shields.io/badge/framework-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/CRF-PyTorch--CRF-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch CRF"/>
  <img src="https://img.shields.io/badge/Python-%3E%3D3.12-blue?style=for-the-badge" alt="Python 3.12+"/>
</p>

A **token-level sequence labeling** system that automatically restores punctuation (commas, periods, and question marks) in unpunctuated Kazakh text. Built on top of `xlm-roberta-large` fine-tuned with a **Conditional Random Field (CRF)** decoder for structured output, and served via a **FastAPI** REST API.

---

## Features

- **Punctuation Restoration** — Predicts commas (`,`), periods (`.`), and question marks (`?`) in raw Kazakh text
- **XLM-RoBERTa + CRF** — Multilingual transformer backbone with a CRF layer for globally consistent label sequences
- **Sliding Window Inference** — Handles long texts beyond the 512-token limit via overlapping window aggregation
- **Rule-based postprocessing** — Kazakh question particles (`ма`, `ме`, `ба`, etc.) are always tagged as `QUESTION`
- **FastAPI backend** — Low-latency REST endpoint ready for integration
- **Custom training pipeline** — Streams data from HuggingFace Datasets (`multidomain-kazakh-dataset`, Kazakh Wikipedia)

---

## Project Structure

```
kazpunct/
├── backend/
│   ├── inference.py        # BestCRFClassifier model definition (XLM-RoBERTa + CRF)
│   └── main.py             # FastAPI application & prediction logic
├── model/
│   ├── data/
│   │   ├── test.csv        # Competition test set
│   │   └── train_example.csv  # Examples included in training
│   ├── best_trained/       # Best saved model weights (model.pt) — not tracked in git
│   ├── trained/            # Latest training output
│   └── train.py            # Full training pipeline
├── pyproject.toml          # Project dependencies (managed with uv)
├── uv.lock                 # Locked dependency versions
└── README.md
```

---

## Model Architecture

```
Input Text (unpunctuated Kazakh)
        │
        ▼
  XLM-RoBERTa-large  ──▶  Token Embeddings (1024-dim)
        │
        ▼
  Dropout (0.1)
        │
        ▼
  Linear Layer  ──▶  Logits (4 classes: O, COMMA, PERIOD, QUESTION)
        │
        ▼
  CRF Decoder  ──▶  Viterbi-decoded label sequence
        │
        ▼
  Post-processing (question particles + capitalization)
        │
        ▼
  Restored Kazakh Sentence
```

### Label Schema

| Label      | Meaning                          | Example suffix |
|------------|----------------------------------|----------------|
| `O`        | No punctuation after this token  | `word`         |
| `COMMA`    | Comma follows this token         | `word,`        |
| `PERIOD`   | Period follows this token        | `word.`        |
| `QUESTION` | Question mark follows this token | `word?`        |

---

## Getting Started

### Prerequisites

- Python **≥ 3.12**
- [`uv`](https://docs.astral.sh/uv/) package manager *(recommended)*, or `pip`
- A trained model checkpoint at `model/best_trained/model.pt`
- GPU with CUDA is optional but recommended for fast inference

### Installation

```bash
# Clone the repository
git clone https://github.com/winux125/Restoration-Kazakh-sentence.git
cd Restoration-Kazakh-sentence

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .
```

### Running the API Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The server will load the model on startup and print `Model ready!` when it's ready to accept requests.

> **Note:** The model checkpoint (`model/best_trained/model.pt`) must exist before starting the server. See [Training](#-training) to generate it, or download a pre-trained checkpoint.

---

## API Reference

### `GET /`

Health check endpoint.

**Response:**
```json
{ "message": "ok" }
```

---

### `POST /restore`

Restores punctuation in unpunctuated Kazakh text.

**Request body:**
```json
{
  "text": "менің атым алихан мен алматыда тұрамын сен қайдан келдің"
}
```

**Response:**
```json
{
  "result": "Менің атым алихан, мен алматыда тұрамын. Сен қайдан келдің?"
}
```

**Example with `curl`:**
```bash
curl -X POST http://localhost:8000/restore \
     -H "Content-Type: application/json" \
     -d '{"text": "менің атым алихан мен алматыда тұрамын сен қайдан келдің"}'
```

---

## Training

The training pipeline in `model/train.py` fine-tunes `xlm-roberta-large` with a CRF head on automatically labeled Kazakh text data.

### Data Sources

| Source | Size | Description |
|--------|------|-------------|
| [`kz-transformers/multidomain-kazakh-dataset`](https://huggingface.co/datasets/kz-transformers/multidomain-kazakh-dataset) | ~200 000 texts | Multi-domain Kazakh corpus |
| [`wikimedia/wikipedia` (kk)](https://huggingface.co/datasets/wikimedia/wikipedia) | ~50 000 texts | Kazakh Wikipedia |
| `model/data/train_example.csv` | — | Competition-provided examples |

Labels are automatically extracted from existing punctuation in the source texts, which are then stripped to create the training input.

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | `xlm-roberta-large` |
| Max sequence length | 512 tokens |
| Batch size | 32 |
| Gradient accumulation | 2 steps |
| Learning rate | 1e-5 |
| Epochs | 5 (early stopping, patience=3) |
| LR scheduler | Cosine |
| Max training samples | 120 000 |

### Run Training

```bash
python model/train.py
```

The best model checkpoint is saved to `model/best_trained/model.pt` based on **macro F1** across COMMA, PERIOD, and QUESTION labels.

---

## How Inference Works

For long texts that exceed the 512-token limit, the system uses a **sliding window** approach:

1. The text is split into words and chunked into overlapping windows of ~256 words
2. Each window is encoded and passed through the model independently
3. Logits for each word are **averaged** across all windows that cover it
4. The CRF decoder runs **once** on the averaged logits to produce the final globally consistent label sequence
5. **Rule-based override**: any word matching a known Kazakh question particle (`ма`, `ме`, `ба`, `бе`, `па`, `пе`, `ше`, `ші`) is forcibly labeled as `QUESTION`
6. Punctuation tokens are appended and the first character of the result is capitalized

---

## Dependencies

| Package | Role |
|---------|------|
| `transformers` | XLM-RoBERTa tokenizer & model |
| `pytorch-crf` | CRF layer for structured decoding |
| `torch` | Deep learning backend |
| `fastapi` + `uvicorn` | REST API server |
| `datasets` | Streaming HuggingFace datasets for training |
| `accelerate` | Distributed/mixed precision training support |
| `scipy` | Numerical utilities |
| `seqeval` | NER-style sequence evaluation metrics |

---

## Evaluation

The model is evaluated using **macro F1** across the three punctuation classes:

```
Metric        Description
──────────────────────────────────────────
macro_f1      Unweighted average F1 across COMMA, PERIOD, QUESTION
f1_comma      F1 for comma prediction
f1_period     F1 for period prediction
f1_question   F1 for question mark prediction
```

Tokens labeled `O` (no punctuation) are excluded from the F1 calculation.

---

## License

This project is open source. See the repository for license details.

---

## Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers) for the `xlm-roberta-large` backbone
- [`kz-transformers`](https://huggingface.co/kz-transformers) for the multidomain Kazakh dataset
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf) for the CRF implementation
