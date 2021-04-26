# NLP
Final project for B9138 - Natural Language Processing
Authors: Eduardo Espino Robles, Meha Sadasivam

A detailed outline of our plan is in the `Project proposal.pdf`

General step-by-step analysis can be tracked in `Final Project Summary Notebook.ipynb`

Packages required:
- `gensim`
- `nltk`
- `numpy`
- `pandas`

Folders:

- `Corpora and Dictionaries`: Contains the final corpora and dictionaries used for each of the themes
- `Dataset Preliminaries`: Contains scripts to filter articles using dataset-specific filters, and a notebook for the high level article count
- `LDA_Outputs`: Contains outputs of the LDA model runs - topic-word probability, yearly average probability for each LDA run
- `Reference`: Contains Dataset documentation
- `Regular LDA Runs`: The notebooks used to run the regular LDA models (most of the dictionary/ corpus pre-procesing was already done in the seqLDA runs)
- `Sequential LDA Runs`: The scripts and notebooks used to run/ tweak sequential LDA models
- `Similarity Model`: Contains the scripts used to tokenize the articles, identify and store the similar words per year
- `Summary Stat Tables`: Contains the article counts for each theme (after using the similarity based filter)

Files:
- `config.py`: Contains some constants used across the repo
- `utilities.py`: Contains utility methods that are used across the repo
- `regime_model.py`: Using the topics of interest's yearly probabilities, run a regime model to forecast returns
