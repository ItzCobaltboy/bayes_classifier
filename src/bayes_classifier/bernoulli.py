from __future__ import annotations

import numpy as np


class BernoulliBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        # Laplace smoothing factor.
        self.alpha = alpha
        # Training state flag.
        self.trained = False

        # Learned class labels.
        self.classes = None
        # Learned Bernoulli probabilities per class.
        self.feature_prob = {}
        # Cached log(p) per class.
        self.log_feature_prob = {}
        # Cached log(1-p) per class.
        self.log_feature_complement = {}
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

        # Convert values to binary presence indicators.
        x = (x > 0).astype(float)
        # Cache shape information.
        n_samples, n_features = x.shape
        self.n_features = n_features

        # Reset learned parameters.
        self.feature_prob = {}
        self.log_feature_prob = {}
        self.log_feature_complement = {}
        self.log_prior = {}

        # Estimate Bernoulli parameters per class.
        for cls in self.classes:
            # Slice class samples.
            # Number of samples in class.
            # Smoothed probability of feature presence.
            x_class = x[y == cls]
            n_class = x_class.shape[0]
            p = (np.sum(x_class, axis=0) + self.alpha) / (n_class + 2.0 * self.alpha)
            # Bound probabilities away from 0/1.
            p = np.clip(p, 1e-12, 1.0 - 1e-12)

            # Store probabilities and caches.
            self.feature_prob[cls] = p
            self.log_feature_prob[cls] = np.log(p)
            self.log_feature_complement[cls] = np.log(1.0 - p)
            self.log_prior[cls] = np.log(n_class / n_samples)

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
        # Convert to binary presence indicators.
        x = (x > 0).astype(float)

        # Accumulate class scores.
        scores = []
        for cls in self.classes:
            # Pull per-class cached log terms.
            log_p = self.log_feature_prob[cls]
            log_one_minus_p = self.log_feature_complement[cls]
            # Bernoulli log-likelihood per sample.
            log_likelihood = np.sum(x * log_p + (1.0 - x) * log_one_minus_p, axis=1)
            # Add class prior.
            scores.append(log_likelihood + self.log_prior[cls])

        # Build score matrix shape: (n_samples, n_classes).
        scores = np.vstack(scores).T
        # Pick class with highest score.
        return self.classes[np.argmax(scores, axis=1)]
