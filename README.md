# FlashCardApp

An app that predicts which words need revision and groups vocabulary by meaning or topic.

Datasets used:
[Duolingo Spaced Repetition](https://www.kaggle.com/datasets/aravinii/duolingo-spaced-repetition-data) (Numerical dataset)
It contains approximately 13 million real learning traces, with historical practice counts, time intervals, and observed recall proportions.

[Tatoeba sentence pairs](https://huggingface.co/datasets/Helsinki-NLP/tatoeba) (Natural-language dataset)
Tatoeba provides sentences and linked translations, including custom exports for selected language pairs.

## Data preparation

- `data_exploration/` – exploratory analysis of both datasets (stage 1).
- `data_preparation/przygotowanie_tatoeba.py` (and the same code as a notebook, `.ipynb`) – cleaning of Tatoeba and creation of the English–Spanish subsets used for embeddings and clustering (KAN-13). Outputs go to `data/processed/` (not versioned).
- `docs/reguly_przetwarzania_tatoeba.md` – preprocessing rules, chosen subsets and clustering evaluation metrics.

```bash
pip install -r data_preparation/requirements.txt
python data_preparation/przygotowanie_tatoeba.py
```
