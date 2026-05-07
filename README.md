# COSC Data Science 1

## Team
- Aqeel Somani, PSID: 2364462
- Santiago Reyes, PSID: 2174589
- Ricardo Trevizo, PSID: 2173424

## Overview
This repository compiles all of the work on the COSC 3337 Project, which includes our executed journals, our Group Project Proposal, and our final report, which shows our findings

### Execution order

1. `DBLP_PRE_EDA.ipynb`
2. `DBLP_Cluster_SemiLearning.ipynb`
3. `DBLP_Classification.ipynb`
4. `DBLP_POST_EDA.ipynb`

The files generated from each step are used in the next step, for the most part.

## Preprocessing — `DBLP_PRE_EDA.ipynb`

All three team members contributed to the preprocessing / pre-EDA stage. This notebook handles raw DBLP loading, cleaning, exploratory checks, and TF-IDF feature construction. It produces the artifacts (`dblp_preprocessed.parquet`, `tfidf_matrix.pkl`, `tfidf_vectorizer.pkl`) that every downstream notebook depends on.

## Clustering — `DBLP_Cluster_SemiLearning.ipynb`




## Clustering - `Classification_Models.ipynb` — 
Loads precomputed TF-IDF features, sweeps Multinomial Naive Bayes / Decision Tree / Random Forest, runs all three hypothesis tests, produces every figure used in the report.### Inputs used

The four large data files used for classification to run properly was:
- `dblp_preprocessed.parquet` — raw paper metadata
- `tfidf_matrix.pkl` — precomputed sparse TF-IDF matrix
- `tfidf_vectorizer.pkl`— fitted TF-IDF vectorizer
- `df_umap_labeled.parquet` — topic labels from the upstream NMF pipeline

### Run the notebook end-to-end (recommended)

Defined in the top cells are the file paths in which we loaded the datasets in. Then run the cells **in order, top to bottom**.

Cell Breakdown:

- Cells 1–5: load data, sanity check, train/test split, helper
- Cell 6: Multinomial Naive Bayes sweep, 4 configs
- Cell 7: Decision Tree sweep, 4 configs — slowest is depth=None
- Cell 7b: checkpoint save
- Cell 8: Random Forest sweep, 4 configs on 50k subsample
- Cell 9: master comparison table
- Cell 10: confusion matrix figure
- Cell 11: per-class F1 figure
- Cell 12: Hypothesis 1 — venue features
- Cell 13: Hypothesis 2 — keyword analysis figure
- Cell 14: Hypothesis 3 — citation analysis figure
- Cell 15: case studies + confusion pairs


## DBLP_POST_EDA.ipynb


## Environment setup

Tested with **Python 3.11** on Windows 11.

Install dependencies:
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib jupyter pyarrow

## Things to watch out for when running

- You need a lot of RAM for most of these notebooks!!! At the very least, 32 GB of RAM is needed for most of the project.
- Some file imports are edited at a later point in some notebooks due to saving, quitting, and then later reloading into the same notebook. If there are errors with execution, it is likely due to that.
- A lot of these notebooks take time to run! The `DBLP_ClusterFail.ipynb` by itself took around an hour of execution time due to the large amount of features, so a high-grade CPU would definitely help.
