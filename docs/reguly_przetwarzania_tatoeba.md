# Reguły przetwarzania zbioru Tatoeba (KAN-13)

Dokument opisuje, jak przygotowujemy zbiór Tatoeba do dalszych zadań:
KAN-14 (embeddingi z wytrenowanego transformera), KAN-15 (K-means) oraz
KAN-16/17 (klastrowanie aglomeracyjne i ocena). Reguły są zaimplementowane w
`data_preparation/przygotowanie_tatoeba.py` (ta sama treść w wersji notatnika:
`data_preparation/przygotowanie_tatoeba.ipynb`).

## 1. Wnioski z analizy eksploracyjnej (etap 1)

Źródło: `data_exploration/analiza_eksploracyjna_tatoeba.py`.

| Obserwacja | Wartość | Konsekwencja |
|---|---|---|
| Liczba zdań w `sentences.csv` | 13 620 824, 428 języków | Czytamy plik porcjami i zostawiamy tylko 2 wybrane języki. |
| Brak kodu języka (`\N`) | 530 wierszy (0,0039%) | Nie da się ich przypisać do pary językowej, więc odpadają (filtr języka). |
| Brak ID, ID ≤ 0, pusty tekst | 0 | Kontrola zostaje w skrypcie (krok 1), na wypadek nowszych eksportów. |
| Identyczne wiersze, powtórzone ID | 0 | Brak działania. |
| Powtórzony tekst w tym samym języku | 185 | Deduplikacja (krok 7), z przepięciem tłumaczeń na zachowane ID. |
| Długość zdania | mediana 31 znaków / 5 segmentów, p99 = 129 znaków / 21 segmentów, max 1500 | Odcinamy skrajnie długie teksty (krok 6): to akapity, nie zdania do fiszek. |
| Najkrótsze rekordy | pojedyncze znaki (`x`, `□`, `네`) | Minimum 2 znaki i co najmniej jedna litera (kroki 3 i 6). |
| `links.csv` | 28 580 234 powiązań, 30 528 (0,11%) wskazuje nieistniejące ID | Osierocone powiązania są liczone w raporcie i odpadają przy łączeniu par. |
| Pokrycie tłumaczeniami (pol→eng) | tylko 57,3% zdań polskich ma tłumaczenie angielskie, 96% ma ≤ 1 tłumaczenie | Do klastrowania bierzemy tylko zdania z tłumaczeniem (fiszka potrzebuje rewersu). |

Analiza Duolingo (`analiza_eksploracyjna_duolingo.py`) pokazuje, że kursy
angielsko-hiszpańskie dominują: es→en (28,3%) i en→es (26,5%) to razem ponad
połowa z 12,85 mln śladów nauki.

## 2. Wybrany podzbiór

**Główny podzbiór: zdania angielskie (`eng`) posiadające tłumaczenie
hiszpańskie (`spa`).** Klastrujemy stronę angielską; hiszpańskie tłumaczenie
jest rewersem fiszki.

Uzasadnienie:

1. Para en↔es jest najliczniejsza w zbiorze Duolingo, więc oba zbiory projektu
   dotyczą tych samych kursów (model powtórek z KAN-6/8-10 i grupy słownictwa z
   KAN-15/16 mogą obsłużyć tę samą talię).
2. Angielski to największy język w Tatoeba (2,04 mln zdań), hiszpański ma
   0,44 mln zdań, czyli ponad 3 razy więcej niż polski (0,14 mln).
3. Wytrenowane modele zdaniowe (KAN-14) są najlepiej sprawdzone dla
   angielskiego; tłumaczenia pozwalają później użyć modelu wielojęzycznego.

**Wariant alternatywny:** `pol`→`eng` (analizowany na etapie 1). Uruchamiamy go
tym samym skryptem: `--jezyk-glowny pol --jezyk-tlumaczenia eng`. Reguły są
identyczne.

Z podzbioru tworzymy trzy zbiory:

| Plik (`data/processed/`) | Zawartość | Użycie |
|---|---|---|
| `pary_eng_spa.tsv` | wszystkie oczyszczone pary `id, zdanie, id_tlumaczenia, tlumaczenie` (zdanie może mieć kilka tłumaczeń) | import fiszek do aplikacji |
| `zbior_klastrowanie_eng_spa.tsv` | jedno zdanie angielskie na wiersz + tłumaczenie podstawowe (najmniejsze ID, czyli najstarsze), `liczba_tlumaczen`, `liczba_znakow`, `liczba_segmentow` | KAN-14 (embeddingi dla całości), KAN-15 (K-means / MiniBatchKMeans na pełnym zbiorze) |
| `probka_klastrowanie_eng_spa.tsv` | losowa próba 20 000 wierszy z powyższego, `random_state = 42` | **wspólny** zbiór do porównania K-means i klastrowania aglomeracyjnego (KAN-15/16/17) |
| `tagi_eng.tsv` (opcja `--tagi`) | tagi Tatoeba dla zdań ze zbioru, bez tagów technicznych (`@...`, `OK`) | słabe etykiety tematyczne do metryk zewnętrznych |
| `raport_czyszczenia_eng_spa.tsv` / `.json` | liczba rekordów przed/po każdym kroku, statystyki źródeł | kontrola i dokumentacja |

Próba ma 20 000 zdań, bo klastrowanie aglomeracyjne wymaga macierzy odległości
O(n²): dla 20 tys. punktów to ok. 1,6 GB (float64, forma skondensowana), co
mieści się w Colabie; dla całego zbioru (setki tysięcy zdań) byłoby to
niewykonalne. Losowanie proste zachowuje rozkład długości zdań.

Danych przetworzonych nie wersjonujemy (katalog `data/` jest w `.gitignore`):
każdy generuje je skryptem. Licencja Tatoeba: CC BY 2.0 FR, wymaga podania źródła.

## 3. Reguły czyszczenia

Kolejność ma znaczenie: deduplikacja jest po normalizacji i filtrach, żeby
zachowana kopia była poprawna.

| Krok | Reguła | Uzasadnienie |
|---|---|---|
| 0 | Zostaw tylko języki pary (`eng`, `spa`); wiersze bez kodu języka odpadają. | Zakres zadania; 530 wierszy bez języka i tak nie da się użyć. |
| 1 | Usuń puste teksty. | Kontrola integralności (w obecnym eksporcie 0). |
| 2 | Normalizacja: Unicode NFC, usunięcie znaków niewidocznych (`U+200B–U+200F`, `U+FEFF` itd.) i sterujących, scalenie wielokrotnych spacji, `strip()`. **Bez** zmiany wielkości liter i bez usuwania interpunkcji. | Ten sam tekst ma mieć jedną reprezentację bajtową. Transformery (KAN-14) korzystają z wielkości liter i interpunkcji, a tekst wyświetlamy też na fiszce. |
| 3 | Usuń zdania bez żadnej litery. | Same liczby/symbole nie niosą słownictwa. |
| 4 | Usuń zdania z URL lub adresem e-mail. | Szum, nie materiał do nauki. |
| 5 | Usuń zdania, w których < 90% liter należy do alfabetu języka (łaciński dla `eng/spa/pol`, cyrylica dla `rus/ukr`). | Wyłapuje źle oznaczony język (np. zdanie po rosyjsku z kodem `eng`). |
| 6 | Zachowaj długość 2–150 znaków i ≤ 25 segmentów (słów). | p99 to 129 znaków / 21 segmentów; dłuższe teksty to akapity (max 1500 znaków) i zniekształcają embeddingi. |
| 7 | Dokładne duplikaty `(język, tekst)`: zostaw najmniejsze ID. | 185 przypadków w EDA. |
| 8 | Prawie-duplikaty: ten sam tekst po `casefold()`, usunięciu interpunkcji i scaleniu spacji (np. `I am a student.` / `i am a student!`): zostaw najmniejsze ID. | Takie zdania dają niemal identyczne embeddingi i sztucznie zawyżają gęstość klastrów. Warianty gramatyczne (np. `Myślałem` / `Myślałam`) **zostają**, bo różnią się słownictwem. |
| 9 | Powiązania: oba kierunki z `links.csv` sprowadzone do pary `(id_eng, id_spa)`, powtórzenia usunięte. | `links.csv` zapisuje każde tłumaczenie dwa razy (A→B i B→A). |
| 10 | ID usuniętych duplikatów (kroki 7–8) przepinamy na ID zachowanego zdania, po czym znowu usuwamy powtórzenia. | Nie tracimy tłumaczeń podpiętych pod duplikat. |
| 11 | Usuń pary, w których którakolwiek strona odpadła w krokach 1–8, oraz powiązania do nieistniejących ID. | Spójność par. |
| 12 | Zbiór do klastrowania: tylko zdania z ≥ 1 tłumaczeniem; tłumaczenie podstawowe = najmniejsze ID. | Fiszka wymaga rewersu; najstarsze tłumaczenie jest zwykle najbardziej sprawdzone; wybór jest deterministyczny. |

Świadomie **nie** stosujemy: lematyzacji, usuwania stop-words, zamiany na małe
litery w zapisanym tekście. Dane trafiają do modelu transformera, który
potrzebuje naturalnego zdania.

## 4. Zasady dla kolejnych etapów (KAN-14)

- Wejściem modelu jest kolumna `zdanie` z `zbior_klastrowanie_eng_spa.tsv`
  (bez dodatkowego przetwarzania). Embeddingi zapisujemy w tej samej kolejności
  co wiersze, razem z kolumną `id`.
- Embeddingi normalizujemy L2. Wtedy odległość euklidesowa jest monotoniczna
  względem odległości kosinusowej, więc K-means i Ward (wymagające metryki
  euklidesowej) działają zgodnie z podobieństwem kosinusowym.
- Embeddingi dla próby wybieramy po `id` z pełnych embeddingów, a nie liczymy
  osobno, żeby oba etapy korzystały z identycznych wektorów.

## 5. Metryki oceny klastrowania (K-means i aglomeracyjne)

### 5.1. Wspólny protokół

Żeby porównanie było uczciwe, obie metody oceniamy:

- na tej samej próbie `probka_klastrowanie_eng_spa.tsv` i tych samych
  embeddingach (L2-znormalizowanych),
- dla tej samej siatki liczby klastrów, np. `k ∈ {5, 10, 15, 20, 30, 40, 50}`,
- z metrykami liczonymi na odległości kosinusowej (`metric="cosine"`),
- z ustalonym ziarnem `random_state = 42` (K-means: dodatkowo `n_init = 10`).

### 5.2. Metryki wewnętrzne (główne, nie wymagają etykiet)

| Metryka | Funkcja | Zakres / kierunek | Co mierzy |
|---|---|---|---|
| **Współczynnik sylwetki** (Silhouette) | `sklearn.metrics.silhouette_score(X, labels, metric="cosine")` | [-1, 1], im wyżej tym lepiej | spójność klastra względem odległości do najbliższego sąsiedniego klastra; **metryka główna** do wyboru `k` i porównania metod |
| **Indeks Daviesa–Bouldina** | `sklearn.metrics.davies_bouldin_score(X, labels)` | ≥ 0, im niżej tym lepiej | średnie podobieństwo każdego klastra do najbardziej podobnego klastra |
| **Indeks Calińskiego–Harabasza** | `sklearn.metrics.calinski_harabasz_score(X, labels)` | ≥ 0, im wyżej tym lepiej | stosunek wariancji międzyklastrowej do wewnątrzklastrowej |

### 5.3. Metryki specyficzne dla metody

- **K-means:** inercja (`KMeans.inertia_`) w funkcji `k` jako metoda łokcia;
  **stabilność**: ARI między podziałami z 5 różnych ziaren (średnia ARI ≥ 0,8
  oznacza stabilne klastry).
- **Klastrowanie aglomeracyjne:** **współczynnik korelacji kofenetycznej**
  (`scipy.cluster.hierarchy.cophenet(Z, pdist(X, "cosine"))`, im bliżej 1 tym
  lepiej) do wyboru metody łączenia (`ward`, `average`, `complete`);
  dendrogram do wizualnej oceny miejsca cięcia. Uwaga: `ward` wymaga metryki
  euklidesowej, co przy embeddingach L2-znormalizowanych jest poprawne.

### 5.4. Metryki zewnętrzne (pomocnicze)

Tatoeba nie ma etykiet tematów. Jako słabych etykiet używamy tagów
(`tagi_eng.tsv`, opcja `--tagi`), tylko dla zdań z dokładnie jednym tagiem
spośród tagów występujących ≥ 30 razy w próbie:

| Metryka | Funkcja | Zakres |
|---|---|---|
| Skorygowany indeks Randa (ARI) | `adjusted_rand_score` | [-0,5, 1], 0 = losowo |
| Znormalizowana informacja wzajemna (NMI) | `normalized_mutual_info_score` | [0, 1] |
| Jednorodność / zupełność / V-measure | `homogeneity_completeness_v_measure` | [0, 1] |

Tagi są niepełne i zaszumione, więc metryki zewnętrzne służą do porównania
metod między sobą, a nie jako ocena bezwzględna.

### 5.5. Porównanie metod i ocena praktyczna

- **Zgodność K-means vs aglomeracyjne:** ARI i NMI między ich podziałami dla
  tego samego `k`.
- **Rozkład rozmiarów klastrów:** min, max, odsetek zdań w największym
  klastrze. Klaster to potencjalna talia fiszek, więc wymagamy min. 20 zdań w
  klastrze i nie więcej niż 20% zdań w jednym klastrze.
- **Ocena jakościowa:** dla 10 losowych klastrów oglądamy 10 zdań najbliższych
  centroidowi i oceniamy spójność tematu w skali 1–3.

### 5.6. Reguła wyboru `k` i metody

1. Odrzuć konfiguracje łamiące warunki rozmiaru klastrów (5.5).
2. Spośród pozostałych wybierz najwyższą sylwetkę; przy różnicy < 0,01
   rozstrzyga niższy indeks Daviesa–Bouldina, a potem mniejsze `k`.
3. Wynik potwierdź łokciem (K-means) i korelacją kofenetyczną (aglomeracyjne).

Przykładowy kod oceny:

```python
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

def ocen_klastrowanie(X, etykiety):
    return {
        "silhouette": silhouette_score(X, etykiety, metric="cosine"),
        "davies_bouldin": davies_bouldin_score(X, etykiety),
        "calinski_harabasz": calinski_harabasz_score(X, etykiety),
    }
```

## 6. Uruchomienie

```bash
pip install -r data_preparation/requirements.txt
python data_preparation/przygotowanie_tatoeba.py            # eng-spa (domyślnie)
python data_preparation/przygotowanie_tatoeba.py --tagi     # + tagi do oceny zewnętrznej
python data_preparation/przygotowanie_tatoeba.py --jezyk-glowny pol --jezyk-tlumaczenia eng
```

Wszystkie progi (`--min-znakow`, `--max-znakow`, `--max-segmentow`,
`--min-udzial-alfabetu`, `--rozmiar-probki`, `--ziarno`) można zmienić z linii
poleceń. Na danych testowych o rozmiarze 3 mln zdań skrypt działał ok. 1 minuty
i zużył 1,5 GB RAM; dla pełnego eksportu (13,6 mln zdań) szacujemy kilka minut
i 2–3 GB RAM, więc działa też w Google Colab (notatnik
`przygotowanie_tatoeba.ipynb`).

Po pierwszym uruchomieniu uzupełnijmy tabelę poniżej liczbami z
`raport_czyszczenia_eng_spa.json`:

| Wielkość | Wartość |
|---|---|
| Zdania `eng` po czyszczeniu | _do uzupełnienia_ |
| Zdania `spa` po czyszczeniu | _do uzupełnienia_ |
| Pary `eng-spa` | _do uzupełnienia_ |
| Zbiór do klastrowania | _do uzupełnienia_ |
| Próba | 20 000 |
