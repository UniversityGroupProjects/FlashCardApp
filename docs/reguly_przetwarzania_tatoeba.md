# Tatoeba preprocessing rules (KAN-13)

This document explains how we clean the Tatoeba dataset and how we will evaluate
the clustering built on top of it. The rules are implemented in
`data_preparation/przygotowanie_tatoeba.py` (the same code is available as a
notebook, `przygotowanie_tatoeba.ipynb`).

The cleaned data is used by:

- KAN-14: sentence embeddings with a pretrained transformer
- KAN-15: K-means clustering
- KAN-16/17: agglomerative clustering and evaluation

## What the exploratory analysis showed

The full export has about 13.6 million sentences in 428 languages. The data is
mostly clean: there are no missing IDs, no empty sentences and no repeated rows.
The few problems we found:

- 530 sentences have no language code. We can't use them, so they are dropped.
- 185 sentences repeat the same text in the same language.
- Most sentences are short (median 31 characters, 5 words), but some are whole
  paragraphs of up to 1500 characters. Some are a single character, like `x`.
- About 0.11% of translation links point to sentences that no longer exist.
- Not every sentence has a translation. For example, only 57% of Polish
  sentences have an English one.

The Duolingo analysis showed that English–Spanish courses (both directions)
make up more than half of all learning records.

## Which subset we use

We use **English sentences that have a Spanish translation**. The English
sentence is the one we embed and cluster, and the Spanish translation is the
back of the flashcard.

Why English–Spanish:

- It is the biggest language pair in the Duolingo data, so both datasets cover
  the same courses.
- English is the largest language in Tatoeba, and Spanish has three times more
  sentences than Polish.
- Pretrained sentence models work best on English.

Polish → English, which we looked at in stage 1, stays available as an
alternative. It runs with the same script and the same rules.

## Cleaning rules

The script keeps only the two chosen languages and then applies these steps in
order:

1. **Normalize the text.** Unicode NFC, remove invisible and control
   characters, collapse repeated spaces, trim. We keep letter case and
   punctuation, because the transformer uses them and the text is shown on the
   flashcard.
2. **Remove unusable sentences:** empty ones, ones without any letters, and
   ones containing a URL or e-mail address.
3. **Remove sentences written in the wrong alphabet.** If less than 90% of the
   letters belong to the expected alphabet (Latin for English and Spanish), the
   language code is probably wrong.
4. **Limit the length** to 2–150 characters and at most 25 words. This covers
   more than 99% of sentences and removes paragraphs and single characters.
5. **Remove duplicates.** First exact duplicates, then near-duplicates that
   differ only in case or punctuation (`I am a student.` and `i am a student!`).
   The oldest sentence (lowest ID) is kept, and translations of the removed
   copies are moved to it.
6. **Build translation pairs.** Links are stored in both directions, so we
   merge them into unique English–Spanish pairs and drop pairs where either
   sentence was removed above.
7. **Keep only sentences with a translation** for clustering. If a sentence has
   several translations, the oldest one is its main translation.

We do not lemmatize, remove stop words or lowercase the text. The transformer
needs natural sentences.

## Output files

All outputs go to `data/processed/`. This folder is in `.gitignore`, so
everyone generates it locally by running the script.

- `pary_eng_spa.tsv`
  - Contents: all clean English–Spanish pairs
  - Used for: flashcards
- `zbior_klastrowanie_eng_spa.tsv`
  - Contents: one row per English sentence with its main translation
  - Used for: embeddings (KAN-14) and K-means on the full data (KAN-15)
- `probka_klastrowanie_eng_spa.tsv`
  - Contents: a random sample of 20,000 rows from the file above (seed 42)
  - Used for: comparing K-means and agglomerative clustering (KAN-15–17)
- `tagi_eng.tsv` (only with `--tagi`)
  - Contents: Tatoeba topic tags
  - Used for: evaluation against tags
- `raport_czyszczenia_eng_spa.tsv` and `.json`
  - Contents: how many records each step removed
  - Used for: checking the cleaning

We need the 20,000-row sample because agglomerative clustering builds a
distance matrix that grows with the square of the data size.

Tatoeba is licensed under CC BY 2.0 FR, so we must credit the source.

## Notes for the embedding step

Feed the `zdanie` column to the model as it is. Save the embeddings in the same
row order, together with the `id` column, and normalize them to unit length.
After that, Euclidean distance behaves like cosine distance, so K-means and
Ward linkage work correctly. For the sample, take the embeddings from the full
set by `id` instead of computing them again.

## How we evaluate the clustering

### Common setup

Both methods are compared on the same 20,000-sentence sample, with the same
embeddings and the same numbers of clusters (for example 5, 10, 15, 20, 30, 40
and 50). We use cosine distance and a fixed random seed of 42.

### Main metrics

These don't need any labels, so they are our main tools. All functions are
from `sklearn.metrics`.

| Metric                  | Function                                       | Range    | Better |
| ----------------------- | ---------------------------------------------- | -------- | ------ |
| Silhouette score        | `silhouette_score(X, labels, metric="cosine")` | -1 to 1  | higher |
| Davies–Bouldin index    | `davies_bouldin_score(X, labels)`              | 0 and up | lower  |
| Calinski–Harabasz index | `calinski_harabasz_score(X, labels)`           | 0 and up | higher |

The silhouette score shows how close sentences are to their own cluster
compared to the nearest other one. It is the main metric for choosing the
number of clusters and comparing methods. Davies–Bouldin shows how similar each
cluster is to its closest neighbour, and Calinski–Harabasz compares the spread
between clusters to the spread inside them.

### Method-specific checks

For K-means we plot inertia against the number of clusters (the elbow method)
and check stability: we run it with 5 different seeds and compare the results
with ARI. An average ARI of 0.8 or more means the clusters are stable.

For agglomerative clustering we use the cophenetic correlation coefficient
(`scipy.cluster.hierarchy.cophenet`) to choose the linkage method (ward,
average or complete). The closer to 1, the better. The dendrogram helps decide
where to cut the tree.

### Comparison with tags

Tatoeba has no real topic labels, but some sentences have tags. For sentences
with exactly one common tag we compare the clusters with the tags:

| Metric                               | Function                             | Range                 |
| ------------------------------------ | ------------------------------------ | --------------------- |
| Adjusted Rand index (ARI)            | `adjusted_rand_score`                | -0.5 to 1, 0 = random |
| Normalized mutual information (NMI)  | `normalized_mutual_info_score`       | 0 to 1                |
| Homogeneity, completeness, V-measure | `homogeneity_completeness_v_measure` | 0 to 1                |

The tags are incomplete, so we only use these numbers to compare the two
methods, not as an absolute score.

### Practical checks

- ARI and NMI between the K-means and agglomerative results for the same
  number of clusters show how much the two methods agree.
- A cluster is a possible flashcard deck, so each cluster should have at least
  20 sentences, and no cluster should hold more than 20% of all sentences.
- For 10 random clusters we read the 10 sentences closest to the centre and
  rate how well they share a topic, from 1 to 3.

### Choosing the final result

First drop the results that break the cluster size limits. From the rest, pick
the one with the highest silhouette score. If two are within 0.01 of each other,
pick the one with the lower Davies–Bouldin index, and then the one with fewer
clusters. Confirm the choice with the elbow plot (K-means) or the cophenetic
correlation (agglomerative).

## How to run

pip install -r data_preparation/requirements.txt
python data_preparation/przygotowanie_tatoeba.py # English–Spanish (default)
python data_preparation/przygotowanie_tatoeba.py --tagi # also save topic tags
python data_preparation/przygotowanie_tatoeba.py --jezyk-glowny pol --jezyk-tlumaczenia eng

All limits (length, alphabet share, sample size, seed) can be changed with
command-line flags. On a 3-million-sentence test file the script took about a
minute and 1.5 GB of RAM. The full export should take a few minutes and 2–3 GB,
so it also runs in Google Colab.

After the first real run, add the final numbers from
`raport_czyszczenia_eng_spa.json` here: how many English and Spanish sentences
are left, how many pairs, and how many sentences are in the clustering set.
