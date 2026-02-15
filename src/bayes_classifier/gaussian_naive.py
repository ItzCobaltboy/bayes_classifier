from __future__ import annotations

import numpy as np


class GaussianNaiveBayesClassifier:
    def __init__(self, var_smoothing: float = 1e-9):
        # Variance floor for numerical stability.
        self.var_smoothing = var_smoothing
        # Training state flag.
        self.trained = False

        # Learned class labels.
        self.classes = None
        # Learned mean per class.
        self.mean = {}
        # Learned variance per class.
        self.var = {}
        # Cached inverse variance per class.
        self.inv_var = {}
        # Cached Gaussian normalization constant per class.
        self.log_norm_const = {}
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
        self.mean = {}
        self.var = {}
        self.inv_var = {}
        self.log_norm_const = {}
        self.log_prior = {}

        # Estimate class-wise independent Gaussian parameters.
        for cls in self.classes:
            # Slice class samples.
            x_class = x[y == cls]
            # Mean per feature.
            mean = np.mean(x_class, axis=0)
            # Variance per feature with smoothing.
            var = np.var(x_class, axis=0) + self.var_smoothing

            # Store parameters and caches.
            self.mean[cls] = mean
            self.var[cls] = var
            self.inv_var[cls] = 1.0 / var
            self.log_norm_const[cls] = -0.5 * np.sum(np.log(2.0 * np.pi * var))
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

        # Accumulate class scores.
        scores = []
        for cls in self.classes:
            # Pull class parameter caches.
            mean = self.mean[cls]
            inv_var = self.inv_var[cls]
            # Start from Gaussian normalization constant.
            log_likelihood = self.log_norm_const[cls]
            # Add quadratic form term.
            log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) * inv_var, axis=1)
            # Add class prior.
            scores.append(log_likelihood + self.log_prior[cls])

        # Build score matrix shape: (n_samples, n_classes).
        scores = np.vstack(scores).T
        # Pick class with highest score.
        return self.classes[np.argmax(scores, axis=1)]
