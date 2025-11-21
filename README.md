# EEG Emotion Recognition with Low-Cost Muse EEG

This repository contains the code for a student research project in which electroencephalography (EEG) recordings obtained with a low-cost Muse headset are used to classify positive, neutral and negative emotional states using supervised machine-learning methods. The work was conducted by **Rumeysa Kaplan** (BSc Psychology, University of Health Sciences) as an initial bridge between psychological training, basic neuroscience, and computational data analysis.

---

## 1. Background and aims

Emotional experience and regulation are central to mental health, yet in both research and clinical practice emotions are usually assessed via self-report or clinical interview. EEG provides a non-invasive measure of brain activity and has been widely used to study affective processes, including frontal asymmetry and other markers of emotional valence. Recently, low-cost wearable EEG devices such as the Muse headset have made it possible to record a small number of channels in non-laboratory settings, raising interest in their potential for emotion recognition and mental-health–related applications.

The aim of this project is to provide a transparent, educational example of how a psychology graduate with no prior programming experience can use an open EEG dataset and standard Python libraries to:

1. Train machine-learning classifiers that distinguish positive, neutral and negative emotional states from four-channel EEG recordings, and  
2. Analyse which EEG features and channels contribute most strongly to the model’s decisions, linking the computational results back to psychological and neuroscientific concepts.

---

## 2. Dataset

All analyses are based on the publicly available **EEG Brainwave Dataset: Feeling Emotions**, created by Jordan J. Bird and colleagues and hosted on Kaggle:

- Dataset page: <https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions>

In the original study, EEG was recorded from **two healthy adults** using a **Muse** headset with four electrodes located approximately at TP9, AF7, AF8 and TP10 (10–20 system), sampled at 256 Hz. Participants viewed audio-visual stimuli designed to induce **positive, neutral and negative** emotional conditions. The dataset providers preprocessed the raw EEG and extracted a large set of numerical features for each short segment of data.

For this project, the pre-extracted feature matrix `emotions.csv` is used:

- Each row corresponds to one EEG segment.
- Approximately **2548 numerical features** are provided per segment, including:
  - Time-domain statistics (mean, minimum, quantile-based measures, higher-order moments),
  - Dispersion measures (e.g., standard deviation),
  - Inter-channel covariance descriptors,
  - Frequency-domain features derived from the fast Fourier transform (FFT).
- An additional column stores the categorical label: **NEGATIVE**, **NEUTRAL** or **POSITIVE**.

The raw dataset itself is **not included** in this repository for licensing and size reasons. Users should download `emotions.csv` directly from Kaggle and place it at:

```text
data/raw/emotions.csv
