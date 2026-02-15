<h1 align="center">Bayes Classifier</h1>

<p align="center">
  Bayesian classifiers for educational ML experiments.
</p>

<p align="center">
  <a href="https://pypi.org/project/bayes-classifier/">
    <img src="https://img.shields.io/pypi/v/bayes-classifier.svg">
  </a>
  <img src="https://img.shields.io/github/license/ItzCobaltboy/bayes_classifier">
  <img src="https://img.shields.io/pypi/pyversions/bayes-classifier">
</p>



A lightweight collection of Bayes classifiers for tabular data.

Included classifiers:

- `MultivariateGaussianBayesClassifier`
- `GaussianNaiveBayesClassifier`
- `MultinomialBayesClassifier`
- `BernoulliBayesClassifier`
- `CategoricalBayesClassifier`
- `PoissonBayesClassifier`

`BayesClassifier` is provided as an alias of `MultivariateGaussianBayesClassifier`.

## Install

```bash
pip install bayes-classifier
```

## Quickstart

```python
import numpy as np
from bayes_classifier import GaussianNaiveBayesClassifier

X = np.array([
    [1.0, 2.0],
    [1.2, 1.8],
    [3.0, 3.2],
    [2.8, 3.1],
])
y = np.array([0, 0, 1, 1])

clf = GaussianNaiveBayesClassifier().fit(X, y)
preds = clf.predict(X)
```

## Classifier Guide

Use the classifier that matches your feature distribution:

- `MultivariateGaussianBayesClassifier`: continuous features with full covariance.
- `GaussianNaiveBayesClassifier`: continuous features with conditional independence assumption.
- `MultinomialBayesClassifier`: non-negative count-like features.
- `BernoulliBayesClassifier`: binary/presence features (inputs are thresholded to 0/1).
- `CategoricalBayesClassifier`: discrete category-valued features.
- `PoissonBayesClassifier`: non-negative count features modeled with Poisson rates.

## API

Each classifier follows the same simple interface:

```python
model = SomeBayesClassifier(...)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
