from __future__ import annotations

import numpy as np


class MultinomialBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        # Laplace smoothing factor.
        self.alpha = alpha
        # Training state flag.
        self.trained = False

        # Learned class labels.
        self.classes = None
        # Learned log P(feature|class) per class.
        self.feature_log_prob = {}
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
        # Multinomial expects non-negative count-like features.
        if np.any(x < 0):
            raise ValueError("Multinomial requires non-negative feature counts.")
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
        self.feature_log_prob = {}
        self.log_prior = {}

        # Estimate multinomial parameters per class.
        for cls in self.classes:
            # Slice class samples.
            x_class = x[y == cls]
            # Total count per feature within class.
            feature_count = np.sum(x_class, axis=0)
            # Total token count in class.
            total_count = np.sum(feature_count)

            # Apply Laplace smoothing.
            smoothed = feature_count + self.alpha
            smoothed_total = total_count + self.alpha * n_features

            # Store log probabilities and prior.
            self.feature_log_prob[cls] = np.log(smoothed / smoothed_total)
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
            raise ValueError("Multinomial requires non-negative feature counts.")

        # Accumulate class scores.
        scores = []
        for cls in self.classes:
            # Dot product of counts with log likelihood per feature.
            log_likelihood = x @ self.feature_log_prob[cls]
            # Add class prior.
            scores.append(log_likelihood + self.log_prior[cls])

        # Build score matrix shape: (n_samples, n_classes).
        scores = np.vstack(scores).T
        # Pick class with highest score.
        return self.classes[np.argmax(scores, axis=1)]
