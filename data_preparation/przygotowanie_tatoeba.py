# -*- coding: utf-8 -*-
# %% [markdown]
# # KAN-13: Czyszczenie i przygotowanie zbioru Tatoeba
#
# Skrypt realizuje reguły opisane w `docs/reguly_przetwarzania_tatoeba.md`.
# Wejście: eksporty Tatoeba (`sentences.tar.bz2`, `links.tar.bz2`, opcjonalnie
# `tags.tar.bz2`). Wyjście: oczyszczone pary zdań i podzbiory do klastrowania
# (KAN-14 embeddingi, KAN-15 K-means, KAN-16/17 klastrowanie aglomeracyjne).
#
# Uruchomienie lokalne:
#
#     python data_preparation/przygotowanie_tatoeba.py                 # eng-spa
#     python data_preparation/przygotowanie_tatoeba.py --jezyk-tlumaczenia pol
#     python data_preparation/przygotowanie_tatoeba.py --tagi          # + tagi
#
# Archiwa są pobierane do `data/raw/`, jeśli ich tam nie ma. Można też położyć
# tam rozpakowane pliki `sentences.csv`, `links.csv`, `tags.csv`.

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
# ## Konfiguracja
#
# Domyślny podzbiór: zdania angielskie (`eng`) posiadające tłumaczenie
# hiszpańskie (`spa`). Uzasadnienie wyboru jest w dokumentacji reguł.

# %%
DOMYSLNA_KONFIGURACJA = {
    "jezyk_glowny": "eng",        # język zdań, które klastrujemy
    "jezyk_tlumaczenia": "spa",   # język tłumaczeń (rewers fiszki)
    "min_znakow": 2,
    "max_znakow": 150,
    "max_segmentow": 25,
    "min_udzial_alfabetu": 0.9,   # min. udział liter w oczekiwanym alfabecie
    "rozmiar_probki": 20_000,     # podzbiór dla klastrowania aglomeracyjnego
    "ziarno": 42,
    "katalog_surowe": "data/raw",
    "katalog_wyniki": "data/processed",
    "tagi": False,
}

URL_EKSPORTOW = "https://downloads.tatoeba.org/exports/"
ROZMIAR_PORCJI = 500_000

# Języki pisane alfabetem łacińskim, dla których sprawdzamy alfabet.
JEZYKI_LACINSKIE = {
    "eng", "spa", "pol", "deu", "fra", "ita", "por", "nld", "tur", "fin",
    "hun", "lit", "epo", "ces", "slk", "swe", "dan", "nob", "ron", "hrv",
}
JEZYKI_CYRYLICKIE = {"rus", "ukr", "bel", "bul", "srp", "mkd"}


def wczytaj_konfiguracje():
    """Konfiguracja z argumentów CLI; ignoruje nieznane argumenty (Colab/Jupyter)."""
    parser = argparse.ArgumentParser(description=__doc__)
    for klucz, wartosc in DOMYSLNA_KONFIGURACJA.items():
        flaga = "--" + klucz.replace("_", "-")
        if isinstance(wartosc, bool):
            parser.add_argument(flaga, action="store_true", default=wartosc)
        else:
            parser.add_argument(flaga, type=type(wartosc), default=wartosc)
    argumenty, _ = parser.parse_known_args()
    return vars(argumenty)


# %% [markdown]
# ## Wczytywanie danych
#
# `sentences.csv` ma ~13,6 mln wierszy (1,9 GB w pamięci), dlatego czytamy go
# porcjami i zatrzymujemy tylko dwa wybrane języki oraz zbiór wszystkich ID
# (potrzebny do kontroli osieroconych powiązań w `links.csv`).

# %%
@contextmanager
def otworz_eksport(nazwa, katalog_surowe):
    """Zwraca uchwyt do `<nazwa>.csv` (rozpakowany plik albo archiwum .tar.bz2)."""
    katalog_surowe = Path(katalog_surowe)
    katalog_surowe.mkdir(parents=True, exist_ok=True)
    plik_csv = katalog_surowe / f"{nazwa}.csv"
    if plik_csv.exists():
        with open(plik_csv, "rb") as plik:
            yield plik
        return

    archiwum = katalog_surowe / f"{nazwa}.tar.bz2"
    if not archiwum.exists():
        print(f"Pobieranie {URL_EKSPORTOW}{archiwum.name} ...")
        urllib.request.urlretrieve(URL_EKSPORTOW + archiwum.name, archiwum)

    with tarfile.open(archiwum, mode="r:bz2") as tar:
        element = next(
            e for e in tar.getmembers()
            if e.isfile() and Path(e.name).name == f"{nazwa}.csv"
        )
        with tar.extractfile(element) as plik:
            yield plik


def czytaj_porcjami(plik, kolumny, typy):
    return pd.read_csv(
        plik, sep="\t", header=None, names=kolumny, dtype=typy,
        quoting=csv.QUOTE_NONE, encoding="utf-8", keep_default_na=False,
        na_values={k: ["", r"\N"] for k in kolumny if k != "zdanie"},
        chunksize=ROZMIAR_PORCJI,
    )


def wczytaj_zdania(jezyki, katalog_surowe):
    wybrane, wszystkie_id = [], []
    statystyki = {"wszystkie_wiersze": 0, "brak_kodu_jezyka": 0, "brak_id": 0}
    with otworz_eksport("sentences", katalog_surowe) as plik:
        for porcja in czytaj_porcjami(
            plik, ["id", "jezyk", "zdanie"],
            {"id": "Int64", "jezyk": "string", "zdanie": "string"},
        ):
            statystyki["wszystkie_wiersze"] += len(porcja)
            statystyki["brak_kodu_jezyka"] += int(porcja["jezyk"].isna().sum())
            statystyki["brak_id"] += int(porcja["id"].isna().sum())
            porcja = porcja.dropna(subset=["id"])
            wszystkie_id.append(porcja["id"].to_numpy("int64"))
            wybrane.append(porcja[porcja["jezyk"].isin(jezyki)])

    zdania = pd.concat(wybrane, ignore_index=True)
    zdania["id"] = zdania["id"].astype("int64")
    zdania["zdanie"] = zdania["zdanie"].fillna("")
    return zdania, pd.Index(np.concatenate(wszystkie_id)), statystyki


# %% [markdown]
# ## Reguły czyszczenia zdań
#
# Każdy krok zapisuje do dziennika liczbę rekordów przed i po, osobno dla
# każdego języka. Dziennik trafia do `raport_czyszczenia.csv`.

# %%
ZNAKI_NIEWIDOCZNE = re.compile(r"[​-‏  ⁠﻿]")
ZNAKI_STERUJACE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
BIALE_ZNAKI = re.compile(r"\s+")
LITERA = re.compile(r"[^\W\d_]")
URL_LUB_EMAIL = re.compile(r"https?://|www\.|\S+@\S+\.\w+", re.IGNORECASE)
INTERPUNKCJA = re.compile(r"[^\w\s]")


def normalizuj_tekst(tekst):
    """NFC, usunięcie znaków sterujących i niewidocznych, scalenie spacji."""
    tekst = unicodedata.normalize("NFC", tekst)
    tekst = ZNAKI_NIEWIDOCZNE.sub("", tekst)
    tekst = ZNAKI_STERUJACE.sub(" ", tekst)
    return BIALE_ZNAKI.sub(" ", tekst).strip()


def udzial_alfabetu(tekst, jezyk):
    """Udział liter należących do oczekiwanego alfabetu języka (1.0 = wszystkie)."""
    if jezyk in JEZYKI_LACINSKIE:
        prefiks = "LATIN"
    elif jezyk in JEZYKI_CYRYLICKIE:
        prefiks = "CYRILLIC"
    else:
        return 1.0
    litery = [z for z in tekst if z.isalpha()]
    if not litery:
        return 0.0
    zgodne = sum(unicodedata.name(z, "").startswith(prefiks) for z in litery)
    return zgodne / len(litery)


def klucz_duplikatu(zdania):
    """Klucz do wykrywania prawie-duplikatów: wielkość liter i interpunkcja pomijane."""
    return (
        zdania.str.casefold()
        .str.replace(INTERPUNKCJA, "", regex=True)
        .str.replace(BIALE_ZNAKI, " ", regex=True)
        .str.strip()
    )


class Dziennik:
    def __init__(self):
        self.wpisy = []

    def filtruj(self, zdania, maska_zachowaj, krok, opis):
        for jezyk, grupa in zdania.groupby("jezyk"):
            zachowane = int(maska_zachowaj.loc[grupa.index].sum())
            self.wpisy.append({
                "krok": krok, "opis": opis, "jezyk": jezyk,
                "przed": len(grupa), "po": zachowane,
                "usunieto": len(grupa) - zachowane,
            })
        return zdania[maska_zachowaj]

    def zapisz(self, krok, opis, jezyk, przed, po):
        self.wpisy.append({
            "krok": krok, "opis": opis, "jezyk": jezyk,
            "przed": przed, "po": po, "usunieto": przed - po,
        })

    def tabela(self):
        return pd.DataFrame(self.wpisy)


def deduplikuj(zdania, kolumna_klucza, krok, opis, dziennik, mapa_id):
    """Zostawia najstarsze zdanie (najmniejsze ID) dla każdego (język, klucz).

    Usunięte ID są mapowane na ID zachowanego zdania, aby nie tracić
    powiązań z `links.csv` wskazujących na duplikat.
    """
    zdania = zdania.sort_values("id")
    kanoniczne = zdania.groupby(["jezyk", kolumna_klucza], sort=False)["id"].transform("first")
    duplikaty = zdania["id"].ne(kanoniczne)
    mapa_id.update(dict(zip(zdania.loc[duplikaty, "id"], kanoniczne[duplikaty])))
    return dziennik.filtruj(zdania, ~duplikaty, krok, opis)


def wyczysc_zdania(zdania, konf, dziennik):
    mapa_id = {}

    zdania = dziennik.filtruj(
        zdania, zdania["zdanie"].str.len().gt(0), "1", "brak tekstu"
    )

    zdania = zdania.copy()
    znormalizowane = zdania["zdanie"].map(normalizuj_tekst)
    for jezyk, grupa in zdania.groupby("jezyk"):
        zmienione = int((znormalizowane[grupa.index] != grupa["zdanie"]).sum())
        dziennik.zapisz("2", f"normalizacja Unicode/spacji (zmieniono {zmienione})",
                        jezyk, len(grupa), len(grupa))
    zdania["zdanie"] = znormalizowane

    zdania = dziennik.filtruj(
        zdania, zdania["zdanie"].str.contains(LITERA), "3",
        "pusty po normalizacji lub bez żadnej litery",
    )
    zdania = dziennik.filtruj(
        zdania, ~zdania["zdanie"].str.contains(URL_LUB_EMAIL), "4",
        "zawiera URL lub e-mail",
    )

    # Alfabet sprawdzamy tylko dla zdań z innymi znakami niż ASCII (szybciej).
    udzial = pd.Series(1.0, index=zdania.index)
    nie_ascii = zdania["zdanie"].str.contains(r"[^\x00-\x7F]")
    udzial[nie_ascii] = [
        udzial_alfabetu(t, j)
        for t, j in zip(zdania.loc[nie_ascii, "zdanie"], zdania.loc[nie_ascii, "jezyk"])
    ]
    zdania = dziennik.filtruj(
        zdania, udzial.ge(konf["min_udzial_alfabetu"]), "5",
        f"udział liter w alfabecie języka < {konf['min_udzial_alfabetu']}",
    )

    zdania = zdania.assign(
        liczba_znakow=zdania["zdanie"].str.len().astype("int32"),
        liczba_segmentow=zdania["zdanie"].str.count(r"\S+").astype("int32"),
    )
    w_zakresie = (
        zdania["liczba_znakow"].between(konf["min_znakow"], konf["max_znakow"])
        & zdania["liczba_segmentow"].le(konf["max_segmentow"])
    )
    zdania = dziennik.filtruj(
        zdania, w_zakresie, "6",
        f"długość poza [{konf['min_znakow']}, {konf['max_znakow']}] znaków "
        f"lub > {konf['max_segmentow']} segmentów",
    )

    zdania = deduplikuj(zdania, "zdanie", "7", "dokładny duplikat (język, tekst)",
                        dziennik, mapa_id)
    zdania = zdania.assign(klucz=klucz_duplikatu(zdania["zdanie"]))
    zdania = deduplikuj(zdania, "klucz", "8",
                        "prawie-duplikat (wielkość liter, interpunkcja)",
                        dziennik, mapa_id)
    return zdania.drop(columns="klucz").reset_index(drop=True), mapa_id


# %% [markdown]
# ## Powiązania (tłumaczenia)
#
# `links.csv` przechowuje powiązania w obu kierunkach. Bierzemy oba kierunki
# (A→B oraz odwrócone B→A), mapujemy ID duplikatów na ID kanoniczne i usuwamy
# powtórzenia. Osobno liczymy powiązania wskazujące nieistniejące ID.

# %%
def wczytaj_pary(ids_glowne_surowe, ids_tlum_surowe, wszystkie_id, katalog_surowe):
    kawalki = []
    statystyki = {"wszystkie_powiazania": 0, "osierocone_powiazania": 0}
    with otworz_eksport("links", katalog_surowe) as plik:
        for porcja in czytaj_porcjami(
            plik, ["id_a", "id_b"], {"id_a": "int64", "id_b": "int64"}
        ):
            statystyki["wszystkie_powiazania"] += len(porcja)
            brak = (wszystkie_id.get_indexer(porcja["id_a"]) == -1) | (
                wszystkie_id.get_indexer(porcja["id_b"]) == -1
            )
            statystyki["osierocone_powiazania"] += int(brak.sum())

            a_glowne = ids_glowne_surowe.get_indexer(porcja["id_a"]) >= 0
            b_glowne = ids_glowne_surowe.get_indexer(porcja["id_b"]) >= 0
            a_tlum = ids_tlum_surowe.get_indexer(porcja["id_a"]) >= 0
            b_tlum = ids_tlum_surowe.get_indexer(porcja["id_b"]) >= 0

            kawalki.append(pd.DataFrame({
                "id": porcja.loc[a_glowne & b_tlum, "id_a"].to_numpy(),
                "id_tlumaczenia": porcja.loc[a_glowne & b_tlum, "id_b"].to_numpy(),
            }))
            kawalki.append(pd.DataFrame({
                "id": porcja.loc[b_glowne & a_tlum, "id_b"].to_numpy(),
                "id_tlumaczenia": porcja.loc[b_glowne & a_tlum, "id_a"].to_numpy(),
            }))
    pary = pd.concat(kawalki, ignore_index=True)
    statystyki["pary_surowe_z_powtorzeniami"] = len(pary)
    return pary, statystyki


def rozwiaz_mape_id(mapa_id):
    """Spłaszcza łańcuchy (duplikat z kroku 7 -> zdanie usunięte w kroku 8 -> ...)."""
    wynik = {}
    for stare in mapa_id:
        nowe = mapa_id[stare]
        while nowe in mapa_id:
            nowe = mapa_id[nowe]
        wynik[stare] = nowe
    return wynik


def wyczysc_pary(pary, zdania, mapa_id, konf, dziennik):
    glowny, tlum = konf["jezyk_glowny"], konf["jezyk_tlumaczenia"]
    para = f"{glowny}-{tlum}"
    przed = len(pary)
    pary = pary.drop_duplicates().copy()
    dziennik.zapisz("9", "powtórzone powiązanie (oba kierunki w links.csv)",
                    para, przed, len(pary))

    mapa = rozwiaz_mape_id(mapa_id)
    for kolumna in ["id", "id_tlumaczenia"]:
        pary[kolumna] = pary[kolumna].map(mapa).fillna(pary[kolumna]).astype("int64")
    przed = len(pary)
    pary = pary.drop_duplicates()
    dziennik.zapisz("10", "powtórzenie po scaleniu duplikatów zdań", para,
                    przed, len(pary))

    zachowane_glowne = set(zdania.loc[zdania["jezyk"].eq(glowny), "id"])
    zachowane_tlum = set(zdania.loc[zdania["jezyk"].eq(tlum), "id"])
    przed = len(pary)
    pary = pary[pary["id"].isin(zachowane_glowne)
                & pary["id_tlumaczenia"].isin(zachowane_tlum)]
    dziennik.zapisz("11", "jedna ze stron usunięta przez reguły 1-8", para,
                    przed, len(pary))

    teksty = zdania.set_index("id")["zdanie"]
    pary = pary.assign(
        zdanie=pary["id"].map(teksty),
        tlumaczenie=pary["id_tlumaczenia"].map(teksty),
    )
    return pary.sort_values(["id", "id_tlumaczenia"]).reset_index(drop=True)


# %% [markdown]
# ## Podzbiory
#
# * `pary_<A>_<B>.tsv` – wszystkie oczyszczone pary zdanie–tłumaczenie.
# * `zbior_klastrowanie_<A>_<B>.tsv` – jedno zdanie główne na wiersz
#   (z tłumaczeniem podstawowym = najstarszym), pełny zbiór do KAN-14/15.
# * `probka_klastrowanie_<A>_<B>.tsv` – losowa próba (ziarno 42), wspólna dla
#   K-means i klastrowania aglomeracyjnego (KAN-15/16/17).

# %%
def zbuduj_podzbiory(pary, zdania, konf):
    statystyki_zdan = zdania.set_index("id")[["liczba_znakow", "liczba_segmentow"]]
    zbior = (
        pary.groupby("id", sort=True)
        .agg(zdanie=("zdanie", "first"),
             id_tlumaczenia=("id_tlumaczenia", "first"),
             tlumaczenie=("tlumaczenie", "first"),
             liczba_tlumaczen=("id_tlumaczenia", "size"))
        .join(statystyki_zdan)
        .reset_index()
    )
    n = min(konf["rozmiar_probki"], len(zbior))
    probka = zbior.sample(n=n, random_state=konf["ziarno"]).sort_values("id")
    return zbior, probka.reset_index(drop=True)


def wczytaj_tagi(ids, katalog_surowe):
    """Tagi Tatoeba jako słabe etykiety tematyczne (ocena zewnętrzna klastrów)."""
    kawalki = []
    with otworz_eksport("tags", katalog_surowe) as plik:
        for porcja in czytaj_porcjami(
            plik, ["id", "zdanie"], {"id": "Int64", "zdanie": "string"}
        ):
            porcja = porcja.dropna(subset=["id"])
            kawalki.append(porcja[porcja["id"].isin(ids)])
    tagi = pd.concat(kawalki, ignore_index=True).rename(columns={"zdanie": "tag"})
    tagi["tag"] = tagi["tag"].str.strip()
    # Tagi techniczne (np. "@needs native check") nie opisują tematu zdania.
    # Tag "OK" oznacza jedynie sprawdzoną poprawność, więc też go pomijamy.
    tagi = tagi[~tagi["tag"].str.startswith("@") & ~tagi["tag"].isin(["", "OK"])]
    return tagi.drop_duplicates().sort_values(["id", "tag"]).reset_index(drop=True)


# %% [markdown]
# ## Uruchomienie

# %%
def zapisz_tsv(df, sciezka):
    df.to_csv(sciezka, sep="\t", index=False, quoting=csv.QUOTE_NONE,
              escapechar="\\", encoding="utf-8")
    print(f"  {sciezka}  ({len(df):,} wierszy)")


def main(konf=None):
    konf = konf or wczytaj_konfiguracje()
    glowny, tlum = konf["jezyk_glowny"], konf["jezyk_tlumaczenia"]
    katalog_wyniki = Path(konf["katalog_wyniki"])
    katalog_wyniki.mkdir(parents=True, exist_ok=True)
    dziennik = Dziennik()

    zdania, wszystkie_id, stat_zdan = wczytaj_zdania([glowny, tlum], konf["katalog_surowe"])
    print("Statystyki sentences.csv:", stat_zdan)

    ids_glowne = pd.Index(zdania.loc[zdania["jezyk"].eq(glowny), "id"])
    ids_tlum = pd.Index(zdania.loc[zdania["jezyk"].eq(tlum), "id"])
    pary, stat_par = wczytaj_pary(ids_glowne, ids_tlum, wszystkie_id, konf["katalog_surowe"])
    print("Statystyki links.csv:", stat_par)
    del wszystkie_id

    zdania, mapa_id = wyczysc_zdania(zdania, konf, dziennik)
    pary = wyczysc_pary(pary, zdania, mapa_id, konf, dziennik)
    zbior, probka = zbuduj_podzbiory(pary, zdania, konf)

    print("Zapisywanie wyników:")
    sufiks = f"{glowny}_{tlum}"
    zapisz_tsv(pary[["id", "zdanie", "id_tlumaczenia", "tlumaczenie"]],
               katalog_wyniki / f"pary_{sufiks}.tsv")
    zapisz_tsv(zbior, katalog_wyniki / f"zbior_klastrowanie_{sufiks}.tsv")
    zapisz_tsv(probka, katalog_wyniki / f"probka_klastrowanie_{sufiks}.tsv")
    if konf["tagi"]:
        zapisz_tsv(wczytaj_tagi(set(zbior["id"]), konf["katalog_surowe"]),
                   katalog_wyniki / f"tagi_{glowny}.tsv")

    tabela = dziennik.tabela()
    zapisz_tsv(tabela, katalog_wyniki / f"raport_czyszczenia_{sufiks}.tsv")
    raport = {
        "konfiguracja": konf,
        "sentences_csv": stat_zdan,
        "links_csv": stat_par,
        "zdania_po_czyszczeniu": zdania["jezyk"].value_counts().to_dict(),
        "pary": len(pary),
        "zbior_klastrowanie": len(zbior),
        "probka_klastrowanie": len(probka),
        "zdania_glowne_bez_tlumaczenia": int(
            zdania["jezyk"].eq(glowny).sum() - len(zbior)
        ),
        "rozklad_liczby_tlumaczen": {
            str(k): int(v) for k, v in zbior["liczba_tlumaczen"].value_counts().sort_index().items()
        },
    }
    with open(katalog_wyniki / f"raport_czyszczenia_{sufiks}.json", "w", encoding="utf-8") as f:
        json.dump(raport, f, ensure_ascii=False, indent=2)

    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print("\nDziennik czyszczenia:")
        print(tabela.to_string(index=False))
        print("\nPrzykładowe rekordy zbioru do klastrowania:")
        print(zbior.head(10).to_string(index=False))
    return raport


if __name__ == "__main__":
    main()
