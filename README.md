![Python](https://img.shields.io/badge/Python-3.14-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-neural%20networks-red)


# IDM Phase-Transition Surrogate

Machine-learning surrogate models for predicting one-step first-order phase-transition observables in the **Inert Doublet Model (IDM)**.

The project studies whether computationally expensive phase-transition observables can be approximated from a small set of BSM model parameters using standard machine-learning and neural-network regression methods. In addition to conventional random train/validation/test splits, the project includes a stricter **region-held-out generalization experiment** designed to test how well the surrogate performs in an unseen part of parameter space.

---

## Overview

The workflow is organized around four Jupyter notebooks and a Python prediction script:

1. **Exploratory analysis**
   Loads and cleans the phase-transition dataset, selects the one-step `oh1` transition sample, and studies the distributions and correlations of the model parameters and phase-transition observables.

2. **Random-split ML benchmark**
   Trains and compares:

   * Linear Regression
   * Histogram Gradient Boosting
   * PyTorch MLP

3. **Region-held-out generalization**
   Repeats the benchmark with a contiguous region of parameter space excluded from training, allowing the project to test a more demanding form of generalization.

4. **Final surrogate training**
   Trains the final PyTorch MLP on the complete one-step transition sample and saves the model, scalers, and metadata needed for inference.

5. **`predict.py`**
   Provides a command-line interface for making predictions from five IDM parameters.

---

## Scientific problem

The dataset contains points in the IDM parameter space together with quantities describing the associated first-order phase transition.

The machine-learning problem is formulated as

$$
(m_H,\;m_A,\;m_{H^\pm},\;\lambda_2,\;\lambda_{345})
\longrightarrow
(T_{\rm crit},\;T_{\rm nuc},\;\alpha,\;\beta/H).
$$

For training, `alpha` and `beta/H` are represented in logarithmic form:

```text
Tcrit
Tnuc
log10(alpha)
log10(beta/H)
```

The final prediction interface converts these transformed quantities back to the physical observables:

```text
Tcrit
Tnuc
alpha
beta/H
```

---

## Dataset

The notebooks use:

```text
../data/idm_pt.tsv
```

The original dataset contains **97,588 rows and 18 columns**. After cleaning and selecting the one-step first-order phase-transition sample (`transition_pattern == "oh1"`), the ML dataset contains **50,297 events**.
The five input features are:

```text
mH
mA
mHp
l2
l345
```

The four ML targets are:

```text
Tcrit
Tnuc
log_alpha
log_beta/H
```

The preprocessing replaces unsuccessful `Tcrit` values of `-999` with the corresponding `Tnuc`, removes samples where `beta/H` could not be computed, and constructs the logarithmic targets.

---

## Project structure

A suggested repository layout is:

```text
.
├── data/
│   └── idm_pt.tsv
│
├── notebooks/
│   ├── notebook_1.ipynb
│   ├── notebook_2.ipynb
│   ├── notebook_3.ipynb
│   └── notebook_4.ipynb
│
├── models/
│   ├── nb2/
│   ├── nb3/
│   └── nb4/
│
├── results/
│   ├── nb2/
│   └── nb3/
│
├── predict.py
├── model_metadata.json
└── README.md
```

The exact notebook filenames can be renamed to make their roles explicit, for example:

```text
01_dataset_audit.ipynb
02_random_split_surrogate.ipynb
03_region_heldout_surrogate.ipynb
04_train_final_surrogate.ipynb
```

---

## Models

Three model classes are compared in the benchmark notebooks.

### 1. Linear Regression

A simple baseline implemented with scikit-learn:

```python
LinearRegression()
```

### 2. Histogram Gradient Boosting

A nonlinear tree-based model using:

```python
HistGradientBoostingRegressor
```

wrapped with `MultiOutputRegressor`.

### 3. PyTorch MLP

The main neural-network surrogate uses the architecture:

```text
5 inputs
  ↓
Linear(5, 64)
  ↓
ReLU
  ↓
Linear(64, 64)
  ↓
ReLU
  ↓
Linear(64, 32)
  ↓
ReLU
  ↓
Linear(32, 4)
```

The network is trained with Adam and mean-squared error loss. The implementation uses standardized input features and standardized targets.

A fixed random seed of `42` is used throughout the ML experiments for reproducibility.

---

## Random train/validation/test benchmark

The one-step dataset is divided into approximately:

```text
70% training
15% validation
15% test
```

corresponding to:

```text
35,207 training samples
7,545 validation samples
7,545 test samples
```

On the random test split, the neural network achieves the following reported performance:

| Target       |   RMSE |    MAE |       R² |
| ------------ | -----: | -----: | -------: |
| `Tcrit`      | 0.2083 | 0.1523 | 0.999823 |
| `Tnuc`       | 0.4480 | 0.2382 | 0.999513 |
| `log_alpha`  | 0.0184 | 0.0100 | 0.999354 |
| `log_beta/H` | 0.0581 | 0.0234 | 0.996680 |

The benchmark also shows the performance of linear regression and histogram gradient boosting, making it possible to compare linear, tree-based, and neural-network approaches.

---

## Region-held-out generalization

A central part of the project is testing whether excellent random-split performance also holds when the model encounters a previously unseen region of parameter space.

The implemented experiment holds out approximately **15% of the data using the upper tail of `mA`**, with a threshold of approximately:

```text
mA = 408.185
```

The held-out region therefore contains values of `mA` approximately between:

```text
408.19 and 499.91
```

while the development region ends at approximately `mA = 408.17`.
This experiment is deliberately more demanding than a conventional random split because the held-out samples occupy a contiguous part of the sampled parameter space.

### Held-out test performance of the MLP

| Target       |   RMSE |    MAE |       R² |
| ------------ | -----: | -----: | -------: |
| `Tcrit`      | 2.1877 | 1.3760 | 0.961380 |
| `Tnuc`       | 5.3957 | 3.0988 | 0.917172 |
| `log_alpha`  | 0.1514 | 0.0797 | 0.901887 |
| `log_beta/H` | 0.1635 | 0.0886 | 0.942074 |

These results show that the apparent accuracy on a random split is not identical to performance in an unseen contiguous region. The notebook explicitly compares the random-test and region-held-out results to quantify this degradation.
This distinction is an important motivation for the project: **high random-test accuracy does not by itself demonstrate robust extrapolation across parameter space.**

---

## Feature importance

The benchmark also evaluates permutation feature importance for the gradient-boosting model.

The results provide a target-dependent measure of how much predictive performance changes when individual BSM parameters are permuted. This is used to study which model parameters contribute most strongly to predictions of `Tcrit`, `Tnuc`, `log_alpha`, and `log_beta/H`.

---

## Final surrogate model

The fourth notebook trains a final PyTorch MLP using the full cleaned one-step FOPT dataset of **50,297 samples**. The final model and preprocessing objects are saved for use outside the notebook.
The final model files are:

```text
models/nb4/final_mlp_regressor.pt
models/nb4/final_feature_scaler.pkl
models/nb4/final_target_scaler.pkl
model_metadata.json
```

The metadata records the model architecture, input and target ordering, transformations, training sample, training size, and feature ranges.

---

## Command-line prediction

After the final model has been trained, predictions can be generated with:

```bash
python predict.py <mH> <mA> <mHp> <l2> <l345>
```

For example:

```bash
python predict.py 180 320 340 2.0 1.0
```

To request JSON output:

```bash
python predict.py 180 320 340 2.0 1.0 --json
```

The script:

1. Loads the trained MLP and scalers.
2. Validates the expected input and target ordering.
3. Scales the five input parameters.
4. Runs the neural network.
5. Transforms the outputs back to `Tcrit`, `Tnuc`, `alpha`, and `beta/H`.
6. Warns if an input lies outside the feature ranges observed during training.

---

## Installation

A typical environment for this project needs:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib torch
```

## The notebooks use NumPy, pandas, Matplotlib, scikit-learn, and PyTorch, with seaborn used for exploratory correlation plots.

## Reproducing the workflow

A typical workflow is:

```text
01. Explore and clean the dataset
        ↓
02. Select one-step FOPT events
        ↓
03. Benchmark ML models on a random split
        ↓
04. Test generalization on a held-out parameter-space region
        ↓
05. Train final MLP on the complete one-step sample
        ↓
06. Save model + scalers + metadata
        ↓
07. Run predictions with predict.py
```

---

## Important notes

This repository focuses specifically on the **one-step `oh1` transition sample** used throughout the ML pipeline.

The region-held-out experiment is intended to probe behavior in an unseen part of the sampled parameter space rather than merely measure interpolation performance under a random split.

The prediction script includes explicit training-domain checks because predictions outside the parameter ranges represented in the training data should be treated with caution.

One implementation detail worth documenting: the prose at the beginning of the region-held-out notebook describes a PCA-based region definition, while the executed split in the notebook is explicitly implemented using the upper 15% of `mA`. For reproducibility, the code-defined `mA` threshold is the operative definition of the reported experiment.

---

## Outputs

The benchmark notebooks save:

```text
results/nb2/
├── validation_results.tsv
├── test_results.tsv
├── permutation_importance.tsv
└── mlp_training_history.tsv
```

The region-held-out experiment additionally saves:

```text
results/nb3/
├── region_validation_results.tsv
├── region_test_results.tsv
├── random_vs_region_test.tsv
├── nn_random_vs_region_degradation.tsv
├── region_mlp_training_history.tsv
└── region_definition.tsv
```

## The notebooks also save the corresponding trained models and scalers.

## Summary

This project demonstrates an end-to-end scientific ML workflow for constructing a surrogate model of IDM first-order phase-transition observables:

* exploratory data analysis and preprocessing;
* comparison of linear, tree-based, and neural-network regressors;
* reproducible train/validation/test evaluation;
* feature-importance analysis;
* explicit testing of generalization to a held-out region of parameter space;
* final model packaging;
* command-line inference with training-domain warnings.

The main purpose is not only to obtain accurate predictions on randomly sampled test points, but also to investigate how surrogate-model accuracy changes when the model is asked to predict in a region of parameter space that was not available during training.
