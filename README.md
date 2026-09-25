# IDM Phase-Transition Surrogate

Machine-learning surrogates for predicting **electroweak phase-transition observables** from parameters of the **Inert Doublet Model (IDM)**.

The project learns the mapping

$$
(m_H, m_A, m_{H^\pm}, \lambda_2, \lambda_{345})
\rightarrow
(T_{\rm crit}, T_{\rm nuc}, \alpha, \beta/H)
$$

for **one-step first-order phase transitions**.

## Models

Three regression approaches are compared:

* Linear Regression
* Histogram Gradient Boosting
* PyTorch MLP

The MLP uses the architecture:

```text
5 → 64 → 64 → 32 → 4
```

with ReLU activations and Adam optimization.

The targets `alpha` and `beta/H` are trained in logarithmic form:

```text
log10(alpha)
log10(beta/H)
```

## Notebooks

**Notebook 1 — Data exploration**

Explores the IDM phase-transition dataset, selects the one-step FOPT sample, examines parameter/observable distributions, and studies correlations.

**Notebook 2 — ML benchmark**

Uses a random train/validation/test split to compare the three models using RMSE, MAE, and \(R^2\), together with prediction plots and permutation feature importance.

**Notebook 3 — Generalization test**

Tests the models on a region of parameter space completely excluded from training. The current implementation holds out the upper 15% of the `mA` distribution.

**Notebook 4 — Final model**

Trains the final MLP on the complete cleaned one-step FOPT dataset and saves the model, scalers, and metadata.

## Outputs

The trained models and preprocessing objects are saved with `joblib` and PyTorch, while evaluation results are stored as `.tsv` files.

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch joblib jupyter
```

## Goal

The aim is to investigate whether ML surrogates can provide a fast approximation to phase-transition observables while remaining reliable beyond simple interpolation within the training sample.
