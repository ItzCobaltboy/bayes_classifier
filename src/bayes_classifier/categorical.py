from __future__ import annotations

import numpy as np


class CategoricalBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        # Additive smoothing factor.
        self.alpha = alpha
        # Training state flag.
        self.trained = False

        # Learned class labels.
        self.classes = None
        # Observed categories per feature index.
        self.feature_categories = {}
        # Nested feature/category log-prob tables per class.
        self.feature_log_prob = {}
        # Learned log prior per class.
        self.log_prior = {}
        # Feature count from fit.
        self.n_features = None

    def fit(self, x_train, y_train):
        # Convert inputs to object arrays for generic categories.
        x = np.asarray(x_train, dtype=object)
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
        self.feature_categories = {}
        self.feature_log_prob = {}
        self.log_prior = {}

        # Record known categories for each feature.
        for j in range(n_features):
            self.feature_categories[j] = np.unique(x[:, j])

        # Estimate per-class categorical probabilities.
        for cls in self.classes:
            # Slice class samples.
            x_class = x[y == cls]
            # Number of samples in class.
            n_class = x_class.shape[0]
            # Allocate nested tables for this class.
            self.feature_log_prob[cls] = {}

            # Build per-feature probability table.
            for j in range(n_features):
                # Known categories for this feature.
                cats = self.feature_categories[j]
                # Initialize counts for all known categories.
                counts = {cat: 0 for cat in cats}
                # Count observed values in the class subset.
                for value in x_class[:, j]:
                    counts[value] = counts.get(value, 0) + 1

                # Number of distinct categories.
                k = len(cats)
                # Smoothed denominator.
                denom = n_class + self.alpha * k

                # Compute log-prob for each known category.
                per_value_log_prob = {}
                for cat in cats:
                    per_value_log_prob[cat] = np.log((counts[cat] + self.alpha) / denom)

                # Fallback log-prob for unseen category at inference.
                unseen_log_prob = np.log(self.alpha / denom)
                # Store feature table.
                self.feature_log_prob[cls][j] = {
                    "known": per_value_log_prob,
                    "unseen": unseen_log_prob,
                }

            # Store class prior.
            self.log_prior[cls] = np.log(n_class / n_samples)

        # Mark model as fitted.
        self.trained = True
        return self

    def predict(self, x_test):
        # Ensure model has been fitted first.
        if not self.trained:
            raise ValueError("Model not trained yet! Call .fit() first")
        assert self.classes is not None and self.n_features is not None

        # Convert test input to object array.
        x = np.asarray(x_test, dtype=object)
        # Promote a single sample to 2D.
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # Validate matrix rank.
        if x.ndim != 2:
            raise ValueError("x_test must be 2D.")
        # Validate feature compatibility.
        if x.shape[1] != self.n_features:
            raise ValueError("x_test must have same number of features as x_train.")

        # Cache test shape.
        n_samples, n_features = x.shape
        # Accumulate class scores.
        scores = []

        # Score each class independently.
        for cls in self.classes:
            # Start from class prior for all samples.
            class_scores = np.full(n_samples, self.log_prior[cls], dtype=float)

            # Add per-feature log-prob contributions.
            for j in range(n_features):
                # Lookup table for known categories.
                known = self.feature_log_prob[cls][j]["known"]
                # Fallback value for unseen categories.
                unseen = self.feature_log_prob[cls][j]["unseen"]
                # Vectorized-ish mapping from values to log-prob.
                class_scores += np.fromiter(
                    (known.get(value, unseen) for value in x[:, j]),
                    dtype=float,
                    count=n_samples,
                )

            # Store class score vector.
            scores.append(class_scores)

        # Build score matrix shape: (n_samples, n_classes).
        scores = np.vstack(scores).T
        # Pick class with highest score.
        return self.classes[np.argmax(scores, axis=1)]
