# -*- coding: utf-8 -*-
# %% [markdown]
# # KAN-13: Cleaning and preparing the Tatoeba dataset
#
# This script applies the rules described in `docs/tatoeba_preprocessing_rules.md`.
# Input: Tatoeba exports (`sentences.tar.bz2`, `links.tar.bz2`, optionally
# `tags.tar.bz2`). Output: cleaned sentence pairs and subsets for clustering
# (KAN-14 embeddings, KAN-15 K-means, KAN-16/17 agglomerative clustering).
#
# Running locally:
#
#     python data_preparation/preparation_of_tatoeba.py                       # eng-spa
#     python data_preparation/preparation_of_tatoeba.py --translation-language pol
#     python data_preparation/preparation_of_tatoeba.py --tags                # + tags
#
# The archives are downloaded to `data/raw/` if they are not there yet. You can
# also put the unpacked files `sentences.csv`, `links.csv`, `tags.csv` there.

# %%
import argparse
import csv
import json
import re
import tarfile
import unicodedata
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

# %% [markdown]
# ## Configuration
#
# Default subset: English sentences (`eng`) that have a Spanish translation
# (`spa`). The reason for this choice is explained in the rules document.

# %%
DEFAULT_CONFIG = {
    "main_language": "eng",          # language of the sentences we cluster
    "translation_language": "spa",   # language of the translations (back of the flashcard)
    "min_chars": 2,
    "max_chars": 150,
    "max_tokens": 25,
    "min_alphabet_share": 0.9,       # min. share of letters in the expected alphabet
    "sample_size": 20_000,           # subset for agglomerative clustering
    "seed": 42,
    "raw_dir": "data/raw",
    "output_dir": "data/processed",
    "tags": False,
}

EXPORTS_URL = "https://downloads.tatoeba.org/exports/"
CHUNK_SIZE = 500_000

# Languages written in the Latin alphabet, for which we check the alphabet.
LATIN_LANGUAGES = {
    "eng", "spa", "pol", "deu", "fra", "ita", "por", "nld", "tur", "fin",
    "hun", "lit", "epo", "ces", "slk", "swe", "dan", "nob", "ron", "hrv",
}
CYRILLIC_LANGUAGES = {"rus", "ukr", "bel", "bul", "srp", "mkd"}


def load_config():
    """Config from CLI arguments; ignores unknown arguments (Colab/Jupyter)."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in DEFAULT_CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true", default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    args, _ = parser.parse_known_args()
    return vars(args)


# %% [markdown]
# ## Loading the data
#
# `sentences.csv` has ~13.6 million rows (1.9 GB in memory), so we read it
# in chunks and keep only the two selected languages plus the set of all IDs
# (needed to count orphaned links in `links.csv`).

# %%
@contextmanager
def open_export(name, raw_dir):
    """Returns a handle to `<name>.csv` (unpacked file or .tar.bz2 archive)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_file = raw_dir / f"{name}.csv"
    if csv_file.exists():
        with open(csv_file, "rb") as file:
            yield file
        return

    archive = raw_dir / f"{name}.tar.bz2"
    if not archive.exists():
        print(f"Downloading {EXPORTS_URL}{archive.name} ...")
        urllib.request.urlretrieve(EXPORTS_URL + archive.name, archive)

    with tarfile.open(archive, mode="r:bz2") as tar:
        member = next(
            m for m in tar.getmembers()
            if m.isfile() and Path(m.name).name == f"{name}.csv"
        )
        with tar.extractfile(member) as file:
            yield file


def read_in_chunks(file, columns, dtypes):
    return pd.read_csv(
        file, sep="\t", header=None, names=columns, dtype=dtypes,
        quoting=csv.QUOTE_NONE, encoding="utf-8", keep_default_na=False,
        na_values={c: ["", r"\N"] for c in columns if c != "sentence"},
        chunksize=CHUNK_SIZE,
    )


def load_sentences(languages, raw_dir):
    selected, all_ids = [], []
    stats = {"total_rows": 0, "missing_language_code": 0, "missing_id": 0}
    with open_export("sentences", raw_dir) as file:
        for chunk in read_in_chunks(
            file, ["id", "lang", "sentence"],
            {"id": "Int64", "lang": "string", "sentence": "string"},
        ):
            stats["total_rows"] += len(chunk)
            stats["missing_language_code"] += int(chunk["lang"].isna().sum())
            stats["missing_id"] += int(chunk["id"].isna().sum())
            chunk = chunk.dropna(subset=["id"])
            all_ids.append(chunk["id"].to_numpy("int64"))
            selected.append(chunk[chunk["lang"].isin(languages)])

    sentences = pd.concat(selected, ignore_index=True)
    sentences["id"] = sentences["id"].astype("int64")
    sentences["sentence"] = sentences["sentence"].fillna("")
    return sentences, pd.Index(np.concatenate(all_ids)), stats


# %% [markdown]
# ## Sentence cleaning rules
#
# Every step writes the number of records before and after to the log,
# separately for each language. The log is saved to
# `cleaning_report_<A>_<B>.tsv`.

# %%
INVISIBLE_CHARS = re.compile(r"[​-‏  ⁠﻿]")
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
WHITESPACE = re.compile(r"\s+")
LETTER = re.compile(r"[^\W\d_]")
URL_OR_EMAIL = re.compile(r"https?://|www\.|\S+@\S+\.\w+", re.IGNORECASE)
PUNCTUATION = re.compile(r"[^\w\s]")


def normalize_text(text):
    """NFC, remove control and invisible characters, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE_CHARS.sub("", text)
    text = CONTROL_CHARS.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def alphabet_share(text, lang):
    """Share of letters that belong to the language's expected alphabet (1.0 = all)."""
    if lang in LATIN_LANGUAGES:
        prefix = "LATIN"
    elif lang in CYRILLIC_LANGUAGES:
        prefix = "CYRILLIC"
    else:
        return 1.0
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    matching = sum(unicodedata.name(ch, "").startswith(prefix) for ch in letters)
    return matching / len(letters)


def duplicate_key(sentences):
    """Key for finding near-duplicates: case and punctuation are ignored."""
    return (
        sentences.str.casefold()
        .str.replace(PUNCTUATION, "", regex=True)
        .str.replace(WHITESPACE, " ", regex=True)
        .str.strip()
    )


class CleaningLog:
    def __init__(self):
        self.entries = []

    def filter(self, sentences, keep_mask, step, description):
        for lang, group in sentences.groupby("lang"):
            kept = int(keep_mask.loc[group.index].sum())
            self.entries.append({
                "step": step, "description": description, "lang": lang,
                "before": len(group), "after": kept,
                "removed": len(group) - kept,
            })
        return sentences[keep_mask]

    def record(self, step, description, lang, before, after):
        self.entries.append({
            "step": step, "description": description, "lang": lang,
            "before": before, "after": after, "removed": before - after,
        })

    def table(self):
        return pd.DataFrame(self.entries)


def deduplicate(sentences, key_column, step, description, log, id_map):
    """Keeps the oldest sentence (smallest ID) for each (language, key).

    Removed IDs are mapped to the ID of the kept sentence, so we do not lose
    links from `links.csv` that point to a duplicate.
    """
    sentences = sentences.sort_values("id")
    canonical = sentences.groupby(["lang", key_column], sort=False)["id"].transform("first")
    duplicates = sentences["id"].ne(canonical)
    id_map.update(dict(zip(sentences.loc[duplicates, "id"], canonical[duplicates])))
    return log.filter(sentences, ~duplicates, step, description)


def clean_sentences(sentences, config, log):
    id_map = {}

    sentences = log.filter(
        sentences, sentences["sentence"].str.len().gt(0), "1", "no text"
    )

    sentences = sentences.copy()
    normalized = sentences["sentence"].map(normalize_text)
    for lang, group in sentences.groupby("lang"):
        changed = int((normalized[group.index] != group["sentence"]).sum())
        log.record("2", f"Unicode/whitespace normalization (changed {changed})",
                   lang, len(group), len(group))
    sentences["sentence"] = normalized

    sentences = log.filter(
        sentences, sentences["sentence"].str.contains(LETTER), "3",
        "empty after normalization or has no letters",
    )
    sentences = log.filter(
        sentences, ~sentences["sentence"].str.contains(URL_OR_EMAIL), "4",
        "contains a URL or e-mail",
    )

    # We check the alphabet only for sentences with non-ASCII characters (faster).
    share = pd.Series(1.0, index=sentences.index)
    non_ascii = sentences["sentence"].str.contains(r"[^\x00-\x7F]")
    share[non_ascii] = [
        alphabet_share(text, lang)
        for text, lang in zip(sentences.loc[non_ascii, "sentence"], sentences.loc[non_ascii, "lang"])
    ]
    sentences = log.filter(
        sentences, share.ge(config["min_alphabet_share"]), "5",
        f"share of letters in the language's alphabet < {config['min_alphabet_share']}",
    )

    sentences = sentences.assign(
        char_count=sentences["sentence"].str.len().astype("int32"),
        token_count=sentences["sentence"].str.count(r"\S+").astype("int32"),
    )
    in_range = (
        sentences["char_count"].between(config["min_chars"], config["max_chars"])
        & sentences["token_count"].le(config["max_tokens"])
    )
    sentences = log.filter(
        sentences, in_range, "6",
        f"length outside [{config['min_chars']}, {config['max_chars']}] characters "
        f"or > {config['max_tokens']} tokens",
    )

    sentences = deduplicate(sentences, "sentence", "7", "exact duplicate (language, text)",
                            log, id_map)
    sentences = sentences.assign(key=duplicate_key(sentences["sentence"]))
    sentences = deduplicate(sentences, "key", "8",
                            "near-duplicate (case, punctuation)",
                            log, id_map)
    return sentences.drop(columns="key").reset_index(drop=True), id_map


# %% [markdown]
# ## Links (translations)
#
# `links.csv` stores links in both directions. We take both directions
# (A→B and the reversed B→A), map duplicate IDs to canonical IDs and remove
# repeats. Separately, we count links that point to IDs that do not exist.

# %%
def load_pairs(main_ids_raw, translation_ids_raw, all_ids, raw_dir):
    parts = []
    stats = {"total_links": 0, "orphaned_links": 0}
    with open_export("links", raw_dir) as file:
        for chunk in read_in_chunks(
            file, ["id_a", "id_b"], {"id_a": "int64", "id_b": "int64"}
        ):
            stats["total_links"] += len(chunk)
            missing = (all_ids.get_indexer(chunk["id_a"]) == -1) | (
                all_ids.get_indexer(chunk["id_b"]) == -1
            )
            stats["orphaned_links"] += int(missing.sum())

            a_main = main_ids_raw.get_indexer(chunk["id_a"]) >= 0
            b_main = main_ids_raw.get_indexer(chunk["id_b"]) >= 0
            a_trans = translation_ids_raw.get_indexer(chunk["id_a"]) >= 0
            b_trans = translation_ids_raw.get_indexer(chunk["id_b"]) >= 0

            parts.append(pd.DataFrame({
                "id": chunk.loc[a_main & b_trans, "id_a"].to_numpy(),
                "translation_id": chunk.loc[a_main & b_trans, "id_b"].to_numpy(),
            }))
            parts.append(pd.DataFrame({
                "id": chunk.loc[b_main & a_trans, "id_b"].to_numpy(),
                "translation_id": chunk.loc[b_main & a_trans, "id_a"].to_numpy(),
            }))
    pairs = pd.concat(parts, ignore_index=True)
    stats["raw_pairs_with_repeats"] = len(pairs)
    return pairs, stats


def resolve_id_map(id_map):
    """Flattens chains (duplicate from step 7 -> sentence removed in step 8 -> ...)."""
    result = {}
    for old in id_map:
        new = id_map[old]
        while new in id_map:
            new = id_map[new]
        result[old] = new
    return result


def clean_pairs(pairs, sentences, id_map, config, log):
    main_lang, trans = config["main_language"], config["translation_language"]
    pair_name = f"{main_lang}-{trans}"
    before = len(pairs)
    pairs = pairs.drop_duplicates().copy()
    log.record("9", "repeated link (both directions in links.csv)",
               pair_name, before, len(pairs))

    mapping = resolve_id_map(id_map)
    for column in ["id", "translation_id"]:
        pairs[column] = pairs[column].map(mapping).fillna(pairs[column]).astype("int64")
    before = len(pairs)
    pairs = pairs.drop_duplicates()
    log.record("10", "repeat after merging duplicate sentences", pair_name,
               before, len(pairs))

    kept_main = set(sentences.loc[sentences["lang"].eq(main_lang), "id"])
    kept_trans = set(sentences.loc[sentences["lang"].eq(trans), "id"])
    before = len(pairs)
    pairs = pairs[pairs["id"].isin(kept_main)
                  & pairs["translation_id"].isin(kept_trans)]
    log.record("11", "one side removed by rules 1-8", pair_name,
               before, len(pairs))

    texts = sentences.set_index("id")["sentence"]
    pairs = pairs.assign(
        sentence=pairs["id"].map(texts),
        translation=pairs["translation_id"].map(texts),
    )
    return pairs.sort_values(["id", "translation_id"]).reset_index(drop=True)


# %% [markdown]
# ## Subsets
#
# * `pairs_<A>_<B>.tsv` – all cleaned sentence–translation pairs.
# * `clustering_set_<A>_<B>.tsv` – one main sentence per row
#   (with the primary translation = the oldest one), the full set for KAN-14/15.
# * `clustering_sample_<A>_<B>.tsv` – random sample (seed 42), shared by
#   K-means and agglomerative clustering (KAN-15/16/17).

# %%
def build_subsets(pairs, sentences, config):
    sentence_stats = sentences.set_index("id")[["char_count", "token_count"]]
    full_set = (
        pairs.groupby("id", sort=True)
        .agg(sentence=("sentence", "first"),
             translation_id=("translation_id", "first"),
             translation=("translation", "first"),
             translation_count=("translation_id", "size"))
        .join(sentence_stats)
        .reset_index()
    )
    n = min(config["sample_size"], len(full_set))
    sample = full_set.sample(n=n, random_state=config["seed"]).sort_values("id")
    return full_set, sample.reset_index(drop=True)


def load_tags(ids, raw_dir):
    """Tatoeba tags as weak topic labels (external evaluation of clusters)."""
    parts = []
    with open_export("tags", raw_dir) as file:
        for chunk in read_in_chunks(
            file, ["id", "sentence"], {"id": "Int64", "sentence": "string"}
        ):
            chunk = chunk.dropna(subset=["id"])
            parts.append(chunk[chunk["id"].isin(ids)])
    tags = pd.concat(parts, ignore_index=True).rename(columns={"sentence": "tag"})
    tags["tag"] = tags["tag"].str.strip()
    # Technical tags (e.g. "@needs native check") do not describe the topic of a sentence.
    # The "OK" tag only means the sentence was checked as correct, so we skip it too.
    tags = tags[~tags["tag"].str.startswith("@") & ~tags["tag"].isin(["", "OK"])]
    return tags.drop_duplicates().sort_values(["id", "tag"]).reset_index(drop=True)


# %% [markdown]
# ## Run

# %%
def save_tsv(df, path):
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE,
              escapechar="\\", encoding="utf-8")
    print(f"  {path}  ({len(df):,} rows)")


def main(config=None):
    config = config or load_config()
    main_lang, trans = config["main_language"], config["translation_language"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log = CleaningLog()

    sentences, all_ids, sentence_stats = load_sentences([main_lang, trans], config["raw_dir"])
    print("sentences.csv stats:", sentence_stats)

    main_ids = pd.Index(sentences.loc[sentences["lang"].eq(main_lang), "id"])
    trans_ids = pd.Index(sentences.loc[sentences["lang"].eq(trans), "id"])
    pairs, link_stats = load_pairs(main_ids, trans_ids, all_ids, config["raw_dir"])
    print("links.csv stats:", link_stats)
    del all_ids

    sentences, id_map = clean_sentences(sentences, config, log)
    pairs = clean_pairs(pairs, sentences, id_map, config, log)
    full_set, sample = build_subsets(pairs, sentences, config)

    print("Saving results:")
    suffix = f"{main_lang}_{trans}"
    save_tsv(pairs[["id", "sentence", "translation_id", "translation"]],
             output_dir / f"pairs_{suffix}.tsv")
    save_tsv(full_set, output_dir / f"clustering_set_{suffix}.tsv")
    save_tsv(sample, output_dir / f"clustering_sample_{suffix}.tsv")
    if config["tags"]:
        save_tsv(load_tags(set(full_set["id"]), config["raw_dir"]),
                 output_dir / f"tags_{main_lang}.tsv")

    table = log.table()
    save_tsv(table, output_dir / f"cleaning_report_{suffix}.tsv")
    report = {
        "config": config,
        "sentences_csv": sentence_stats,
        "links_csv": link_stats,
        "sentences_after_cleaning": sentences["lang"].value_counts().to_dict(),
        "pairs": len(pairs),
        "clustering_set": len(full_set),
        "clustering_sample": len(sample),
        "main_sentences_without_translation": int(
            sentences["lang"].eq(main_lang).sum() - len(full_set)
        ),
        "translation_count_distribution": {
            str(k): int(v) for k, v in full_set["translation_count"].value_counts().sort_index().items()
        },
    }
    with open(output_dir / f"cleaning_report_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print("\nCleaning log:")
        print(table.to_string(index=False))
        print("\nSample records from the clustering set:")
        print(full_set.head(10).to_string(index=False))
    return report


if __name__ == "__main__":
    main()
