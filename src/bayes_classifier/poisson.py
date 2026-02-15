from __future__ import annotations

import numpy as np


class PoissonBayesClassifier:
    def __init__(self, alpha: float = 1e-9):
        # Tiny offset to avoid log(0).
        self.alpha = alpha
        # Training state flag.
        self.trained = False

        # Learned class labels.
        self.classes = None
        # Learned Poisson rate per class.
        self.rate = {}
        # Cached log(rate) per class.
        self.log_rate = {}
        # Learned log prior per class.
        self.log_prior = {}
        # Feature count from fit.
        self.n_features = None

    def fit(self, x_train, y_train):
        # Convert inputs to NumPy arrays.
        x = np.asarray(x_train, dtype=float)
        y = np.asarray(y_train)

        # Validate feature matrix rank.
        if x.ndim != 2:
            raise ValueError("x_train must be 2D.")
        # Poisson expects non-negative counts.
        if np.any(x < 0):
            raise ValueError("Poisson requires non-negative count features.")
        # Validate label vector rank.
        if y.ndim != 1:
            raise ValueError("y_train must be 1D.")
        # Validate sample count match.
        if x.shape[0] != y.shape[0]:
            raise ValueError("x_train and y_train must have same number of rows.")

        # Collect class labels.
        self.classes = np.unique(y)
        # Require at least binary classification.
        if self.classes.size < 2:
            raise ValueError("At least 2 classes are required.")

        # Cache shape information.
        n_samples, n_features = x.shape
        self.n_features = n_features

        # Reset learned parameters.
        self.rate = {}
        self.log_rate = {}
        self.log_prior = {}

        # Estimate class-wise Poisson rates.
        for cls in self.classes:
            # Slice class samples.
            x_class = x[y == cls]
            # Mean count per feature (MLE) with tiny offset.
            lam = np.mean(x_class, axis=0) + self.alpha
            # Guard against zeros.
            lam = np.clip(lam, self.alpha, None)

            # Store parameters and caches.
            self.rate[cls] = lam
            self.log_rate[cls] = np.log(lam)
            self.log_prior[cls] = np.log(x_class.shape[0] / n_samples)

        # Mark model as fitted.
        self.trained = True
        return self

    def predict(self, x_test):
        # Ensure model has been fitted first.
        if not self.trained:
            raise ValueError("Model not trained yet! Call .fit() first")
        assert self.classes is not None and self.n_features is not None

        # Convert test input to NumPy array.
        x = np.asarray(x_test, dtype=float)
        # Promote a single sample to 2D.
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Validate matrix rank.
        if x.ndim != 2:
            raise ValueError("x_test must be 2D.")
        # Validate feature compatibility.
        if x.shape[1] != self.n_features:
            raise ValueError("x_test must have same number of features as x_train.")
        # Enforce non-negative counts.
        if np.any(x < 0):
            raise ValueError("Poisson requires non-negative count features.")

        # Accumulate class scores.
        scores = []
        for cls in self.classes:
            # Poisson log-likelihood without log(x!) constant.
            log_likelihood = np.sum(x * self.log_rate[cls] - self.rate[cls], axis=1)
            # Add class prior.
            scores.append(log_likelihood + self.log_prior[cls])

        # Build score matrix shape: (n_samples, n_classes).
        scores = np.vstack(scores).T
        # Pick class with highest score.
        return self.classes[np.argmax(scores, axis=1)]
