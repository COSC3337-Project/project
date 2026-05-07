# COSC Data Science 1

## Folder Contents
DBLP_Classification.ipynb
DBLP_ClusterFail.ipynb
DBLP_Cluster_SemiLearning.ipynb
DBLP_POST_EDA.ipynb
DBLP_PRE_EDA.ipynb
DBLP_Preprocessing.ipynb
README.md - this file you are currently reading for documentation of the project
Report.pdf - Final report discussing project results
Group_Project_Proposal.pdf - Proposal document 
requirements.txt - all dependencies and packages needed for the attached notebooks 

## Team
- Aqeel Somani, PSID: 2364462
- Santiago Reyes, PSID: 2174589
- Ricardo Trevizo, PSID: 2173424

## Overview
This repository compiles all of the work on the COSC 3337 Project, which includes our executed journals, our Group Project Proposal, and our final report, which shows our findings.

### Execution order
1. `DBLP_PREPROCESSING.ipynb`
2. `DBLP_PRE_EDA.ipynb`
3. `DBLP_Cluster_SemiLearning.ipynb`
4. `DBLP_Classification.ipynb`
5. `DBLP_POST_EDA.ipynb`
The files generated from each step are used in the next step, for the most part.

## Preprocessing — `DBLP_PREPROCESSING.ipynb`
All three team members contributed to the preprocessing / pre-EDA stage. This notebook handles raw DBLP loading, cleaning, exploratory checks, and TF-IDF feature construction. It produces (`dblp_preprocessed.parquet`, `tfidf_matrix.pkl`, `tfidf_vectorizer.pkl`) that every downstream notebook depends on.

## EDA - `DBLP_PRE_EDA.ipynb`
This file reads and loads the `dblp_preprocessed.parquet`, and produces a variety of different graphs to visualize the patterns in the preprocessed data.
- To run, make sure the file path under "SAVEDIR" is correct to load in the `dblp_preprocessed.parquet` file. 

## Clustering — `DBLP_Cluster_SemiLearning.ipynb`
The Cluster/SemiLearning notebook depends on mostly the `tfidf_matrix.pkl` for clustering
and the `tfidf_vectorizer.pkl` for data interpretation. It also uses the
`dblp_preprocessed.parquet` for creating the final `df_umap_labeled.parquet` and
`topic_mapping.json`, which is then used in the classification step.

## Classification - `Classification_Models.ipynb` — 
Loads precomputed TF-IDF features, sweeps Multinomial Naive Bayes / Decision Tree / Random Forest, runs all three hypothesis tests, produces every figure used in the report.### Inputs used

The four large data files used for classification to run properly was:
- `dblp_preprocessed.parquet` — raw paper metadata
- `tfidf_matrix.pkl` — precomputed sparse TF-IDF matrix
- `tfidf_vectorizer.pkl`— fitted TF-IDF vectorizer
(the three files above are generated from DBLP_PREPROCESSING.ipynb)
- `df_umap_labeled.parquet` — topic labels generated from the DBLP_Cluster_SemiLearning.ipynb notebook 

## DBLP_POST_EDA.ipynb

The post-EDA used most of the previous generated datasets and models. The needed files are:

1. df_umap_labeled.parquet
2. papers.parquet
3. tfidf_matrix.pkl
4. tfidf_vectorizer.pkl
5. topic_mapping.json

## Things to watch out for when running
- You need a lot of RAM for most of these notebooks!!! At the very least, 32 GB of RAM is needed for most of the project.
- Some file imports are edited at a later point in some notebooks due to saving, quitting, and then later reloading into the same notebook. If there are errors with execution, it is likely due to that.
- A lot of these notebooks take time to run! The `DBLP_ClusterFail.ipynb` by itself took around an hour of execution time due to the large amount of features, so a high-grade CPU would definitely help.
