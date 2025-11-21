# EEG Emotion Recognition with Low-Cost Muse EEG

This repository contains the code for a student project by **Rumeysa Kaplan** (Psychology BSc, University of Health Sciences), combining psychology, EEG and basic machine learning.

The goal of the project is to classify **positive, neutral and negative** emotional states from EEG signals recorded with a low-cost **Muse** headset, using the publicly available **EEG Brainwave Dataset: Feeling Emotions**.

---

## 1. Dataset

- Dataset: **EEG Brainwave Dataset: Feeling Emotions** by Jordan J. Bird et al.
- Source: Kaggle  
  https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions

The dataset provides:
- 4-channel EEG (Muse: TP9, AF7, AF8, TP10)
- Segments labelled as **NEGATIVE**, **NEUTRAL** or **POSITIVE**
- A pre-computed feature matrix `emotions.csv` with ~2548 features per segment

> Note: The raw data file `emotions.csv` is **not** included in this repository.  
> Please download it yourself from Kaggle and place it under:  
> `data/raw/emotions.csv`

---

## 2. Project structure

```text
eeg-feeling-emotions-ml/
  data/
    processed/              # saved plots and importance tables
  src/
    dataset_emotions.py     # load dataset, encoding, train–test split, scaling
    train_rf.py             # train & evaluate Random Forest classifier
    train_mlp.py            # train & evaluate MLP neural network (PyTorch)
    models_mlp.py           # MLP model and training utilities
    analyze_rf_channels.py  # feature importance: per feature, channel, type
    analyze_rf_importances.py (optional)
  .gitignore
  README.md
