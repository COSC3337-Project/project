# Preprocessing Overview

## What This Notebook Does

The goal of this notebook is to take the raw DBLP academic paper dataset and clean it up so it is ready for EDA, clustering, and classification. The dataset has 3,079,007 paper records across fields like title, abstract, authors, references, venue, and year. The notebook handles missing values, cleans the text, filters out low-quality venues, builds new numerical features, normalizes everything, and produces a TF-IDF word-feature matrix. All outputs are saved to disk so teammates do not need to rerun this notebook.

---

## What is Parquet and PyArrow

Parquet is a file format designed for storing large datasets efficiently. Unlike a CSV or JSON file (which are just plain text), Parquet compresses the data column by column and stores type information alongside it. This makes it much faster to read back and smaller on disk.

PyArrow is the library that handles the actual reading and writing of Parquet files. You do not call it directly — when you run `pd.read_parquet()` or `df.to_parquet()` in pandas, PyArrow is working under the hood automatically.

Why we used it: the original DBLP data comes as several large JSON files that take a few minutes to load every time. By saving to Parquet once, every future run loads all 3 million rows in a few seconds instead.

---

## Missing Values Found

Before any cleaning, we checked every column for missing entries:

| Column | Missing | % of Total |
|---|---|---|
| abstract | 530,475 | 17.23% |
| references | 362,865 | 11.79% |
| authors | 4 | ~0% |
| everything else | 0 | 0% |

About 1 in 6 papers had no abstract, and roughly 1 in 9 had no references list. All other columns were complete.

---

## Preprocessing Steps

### Step 1 – Fill Missing Values

Missing abstracts were filled with an empty string so downstream text processing still works on those rows without crashing. Missing references and authors were filled with empty lists for the same reason. We also added a `has_abstract` column (True/False) so later steps can easily filter to only papers that have an abstract when needed.

### Step 2 – Clean Text

The `title` and `abstract` columns were cleaned by converting everything to lowercase, removing special characters, and collapsing extra whitespace. This produced two new columns — `clean_title` and `clean_abstract` — which were then joined into a single `text_combined` column. Combining title and abstract gives the TF-IDF model more context about each paper's topic.

The cleaning was done in chunks of 50,000 rows at a time to avoid memory issues on a 3 million row dataset.

### Step 3 – Filter Venues

Venues with fewer than 50 papers were removed. A venue with only 1 or 2 papers does not have enough data to be meaningful for clustering or classification. Out of 5,079 unique venues, 2,673 passed the threshold. This removed 15,803 papers, leaving 3,063,204.

### Step 4 – Feature Engineering

The dataset only comes with `year` and `n_citation` as numeric columns. We derived four more:

- `author_count` — how many authors are listed on the paper
- `reference_count` — how many papers it cites
- `abstract_length` — word count of the cleaned abstract
- `decade` — the year rounded down to the nearest decade (e.g. 2013 → 2010)

These give the EDA and clustering steps more signals to work with beyond just citations.

### Step 5 – Normalization

All five numeric columns (`year`, `n_citation`, `author_count`, `reference_count`, `abstract_length`) were checked for missing values (none were found at this point) and then Z-score normalized into new `_z` columns. Z-score normalization rescales each column so it has a mean of 0 and a standard deviation of 1. This prevents columns with larger raw values (like `n_citation`) from dominating models that are sensitive to scale. The original columns are kept so nothing is lost.

### Step 6 – TF-IDF Vectorization

TF-IDF converts the `text_combined` text into a numeric matrix where each row is a paper and each column is a word. The score for each word reflects how important it is to that paper relative to the whole dataset — common words like "the" score low, distinctive words score high.

We used only papers that have an abstract (about 2.5 million), but capped at 300,000 to prevent memory crashes. Settings used:
- 5,000 word features (top most informative words)
- Words appearing in fewer than 5 papers removed (likely typos)
- Words appearing in more than 85% of papers removed (too common to be useful)
- English stop words removed

The result is a 300,000 × 5,000 matrix saved as `tfidf_matrix.pkl`.

---

## Output Files

Three files are saved at the end so teammates can load the preprocessed data directly:

- `dblp_preprocessed.parquet` — the full cleaned DataFrame with all engineered features (3,063,204 rows)
- `tfidf_matrix.pkl` — the TF-IDF sparse matrix (300,000 papers × 5,000 word features)
- `tfidf_vectorizer.pkl` — the fitted vectorizer, needed if you want to transform new text using the same vocabulary

---

# Preprocessing Rationale

## Why We Did Not Use Interpolation

We did not apply linear interpolation because the DBLP dataset is not a naturally ordered time-series where values should change smoothly from one row to the next.

Interpolation is best when records are sequential and neighboring values are meaningfully related (for example, sensor readings over time). In this project, each row is an independent paper record, so interpolating across papers could create artificial values that do not reflect real papers.

This is especially problematic for mixed data types in this dataset:
- Text/list fields (`title`, `abstract`, `authors`, `references`) are not suitable for interpolation.
- Count-like fields (for example, `n_citation`, engineered counts) are often skewed and can be distorted by linear interpolation.

## What We Used Instead

We used methods aligned with the project scope and data structure:
- `abstract` missing values -> empty string.
- `authors` and `references` missing values -> empty list.
- Numeric features used for modeling -> median imputation.
- Numeric scaling for modeling -> Z-score normalization.

## Why This Is Better for This Project

- It preserves realistic record-level semantics.
- It avoids introducing fabricated trends between unrelated papers.
- It supports downstream clustering and classification with cleaner, standardized numeric features.
