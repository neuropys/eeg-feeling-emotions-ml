````markdown
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
````

This project constitutes a **secondary analysis** of anonymised, publicly released data; no new data were collected.

---

## 3. Methods overview

All code is written in **Python** and was developed in **PyCharm**. The analysis pipeline can be summarised as follows:

1. **Data loading and preprocessing**

   * `emotions.csv` is loaded into a pandas DataFrame.
   * The feature matrix (**X**) is constructed by selecting all numerical feature columns; the label vector (**y**) is obtained from the emotion label column.
   * String labels (NEGATIVE, NEUTRAL, POSITIVE) are encoded as integer class indices using a label encoder.
   * A **stratified 80/20 train–test split** is applied to preserve class balance across splits.
   * Features are standardised using `StandardScaler` (zero mean, unit variance), fitted on the training set and applied to both training and test data.

2. **Random Forest classifier** (`src/train_rf.py`)

   * A **RandomForestClassifier** from scikit-learn is trained on the standardised training data.
   * The model uses **200 decision trees**, with other hyperparameters kept close to default values.
   * Performance is evaluated on the test set using overall accuracy, per-class precision, recall and F1-score, and a confusion matrix.

3. **Multilayer perceptron (MLP) neural network** (`src/train_mlp.py`, `src/models_mlp.py`)

   * A shallow feed-forward neural network is implemented in **PyTorch**.
   * Input layer size equals the number of features (2548).
   * Two hidden layers with 256 and 128 units respectively use ReLU activations and dropout (0.3) for regularisation.
   * The output layer has three units corresponding to the three emotion classes; training uses cross-entropy loss and the Adam optimiser.
   * The network is trained for a fixed number of epochs, with monitoring of training and validation losses and final evaluation on the test set using the same metrics as for the Random Forest.

4. **Feature importance analysis** (`src/analyze_rf_channels.py`, `src/analyze_rf_importances.py`)

   * The Random Forest model’s `feature_importances_` attribute is extracted to quantify the contribution of each input feature.
   * A ranked table of features and their importance values is produced, and the top features are visualised.
   * Feature names are parsed to group importances by:

     * **Feature type** (e.g., `mean`, `min`, `covmat`, `fft`), and
     * **Channel code** (suffixes such as `_a`, `_b`, `_c`, `_d` corresponding to muse electrodes).
   * Aggregated importance per feature type and per channel group is computed and plotted, providing a higher-level interpretation of which categories of descriptors and which electrode groups are most informative.

The goal of this methodological design is not to push model performance to a theoretical maximum, but to implement a clear and reproducible analysis that is understandable for readers without an extensive background in computer science.

---

## 4. Repository structure

```text
eeg-feeling-emotions-ml/
  data/
    processed/
      rf_channel_importance.png         # importance by channel group
      rf_feature_importances_top20.png  # top-20 feature importances
      rf_featuretype_importance.png     # importance by feature type
  src/
    dataset_emotions.py                 # data loading, encoding, train–test split, scaling
    train_rf.py                         # Random Forest training and evaluation
    models_mlp.py                       # PyTorch MLP model definition and training utilities
    train_mlp.py                        # MLP training and evaluation
    analyze_rf_channels.py              # aggregate RF importances by channel and feature type
    analyze_rf_importances.py           # detailed RF feature-importance analysis (optional)
  .gitignore
  README.md
```

The virtual environment directory and raw data files are intentionally excluded from version control.

---

## 5. Usage

1. **Clone the repository**

```bash
git clone https://github.com/neuropys/eeg-feeling-emotions-ml.git
cd eeg-feeling-emotions-ml
```

2. **Set up a virtual environment and install dependencies**

A minimal `requirements.txt` would include:

```text
pandas
numpy
scikit-learn
matplotlib
torch
```

Example setup:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

3. **Download the dataset**

* Download `emotions.csv` from Kaggle (see link in Section 2).
* Place it under:

```text
data/raw/emotions.csv
```

4. **Run experiments**

* Random Forest:

```bash
python src/train_rf.py
```

* MLP neural network:

```bash
python src/train_mlp.py
```

* Feature-importance analysis:

```bash
python src/analyze_rf_channels.py
```

Results (classification reports and confusion matrices) are printed to the console; importance tables and plots are saved under `data/processed/`.

---

## 6. Summary of results

On the three-class classification task (NEGATIVE, NEUTRAL, POSITIVE), using a stratified 80/20 train–test split:

* The **Random Forest** model achieved an accuracy of approximately **0.99** and a macro-averaged F1-score of **0.99**, with very few confusions between negative and positive states and almost perfect performance on neutral segments.
* The **MLP** achieved an accuracy of approximately **0.97** and a macro-averaged F1-score of **0.97**, slightly below the Random Forest but still in the high-performance range.

Feature-importance analysis of the Random Forest indicated that:

* Features associated with two of the four Muse channels accounted for the vast majority of overall importance, suggesting a particularly strong contribution of a subset of electrodes (likely including frontal sites).
* The most influential feature categories were **minimum/quantile-based statistics**, **mean-based statistics**, and **inter-channel covariance descriptors**, with FFT-based features contributing a smaller but non-trivial proportion.

These findings are broadly consistent with the affective neuroscience literature emphasising frontal EEG involvement in emotional valence and with previous work on this dataset using tree-based and neural network models.

---

## 7. Ethical and licensing considerations

The project uses exclusively **anonymised, publicly available data** from Kaggle. No new data were collected and no identifiable information is present in the analysed files. The original responsibility for informed consent and ethical approval lies with the dataset creators. This repository distributes only analysis code and derived plots; users are expected to obtain the original dataset directly from Kaggle and abide by its licensing terms.

This work is intended for educational and research purposes only. It is not a medical device and should not be used for clinical decision-making.

---

## 8. Author and acknowledgement

This project was developed by **Rumeysa Kaplan**, a psychology graduate interested in pursuing further study in neuroscience and mental health research. The repository demonstrates how open EEG datasets and standard Python tools can be used by students from non-computational backgrounds to begin working with brain data and machine-learning methods.

The author gratefully acknowledges the creators of the **EEG Brainwave Dataset: Feeling Emotions** for making their data publicly available, and the open-source software community for providing the tools that made this analysis possible.

---

## 9. Citation

If you use this code or structure in your own work, please cite the original dataset as:

> Bird, J. J. (n.d.). *EEG Brainwave Dataset: Feeling Emotions*. Kaggle.
> Retrieved from [https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions)

When appropriate, you may also cite this repository in the form:

> Kaplan, R. (2025). *EEG Emotion Recognition with Low-Cost Muse EEG* (GitHub repository).
> Available at: [https://github.com/neuropys/eeg-feeling-emotions-ml](https://github.com/neuropys/eeg-feeling-emotions-ml)

```
::contentReference[oaicite:0]{index=0}
```
