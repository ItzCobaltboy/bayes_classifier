from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal


class MultivariateGaussianBayesClassifier:
    def __init__(self, reg_covar: float = 1e-6):
        # Small diagonal regularizer for covariance stability.
        self.reg_covar = reg_covar
        # Training state flag.
        self.trained = False

        # Learned class labels.
        self.classes = None
        # Learned mean vector per class.
        self.mean = {}
        # Learned covariance matrix per class.
        self.cov = {}
        # Learned log prior probability per class.
        self.log_prior = {}
        # Feature count seen during fit.
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

        # Collect sorted unique class labels.
        self.classes = np.unique(y)
        # Require at least binary classification.
        if self.classes.size < 2:
            raise ValueError("At least 2 classes are required.")

        # Cache shape information.
        n_samples, n_features = x.shape
        self.n_features = n_features

        # Reset learned parameters.
        self.mean = {}
        self.cov = {}
        self.log_prior = {}

        # Precompute identity matrix once for regularization.
        eye = np.eye(n_features)
        # Estimate parameters independently for each class.
        for cls in self.classes:
            # Slice all samples belonging to the class.
            x_class = x[y == cls]
            # Estimate class mean.
            self.mean[cls] = np.mean(x_class, axis=0)
            # Estimate class covariance.
            cov = np.cov(x_class, rowvar=False, ddof=0)
            # Handle 1D edge case returned as scalar.
            if np.ndim(cov) == 0:
                cov = np.array([[float(cov)]])
            # Add diagonal regularization.
            cov = cov + eye * self.reg_covar
            # Store covariance.
            self.cov[cls] = cov
            # Store class prior in log-space.
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
        # Validate feature compatibility with training.
        if x.shape[1] != self.n_features:
            raise ValueError("x_test must have same number of features as x_train.")

        # Accumulate class scores (log posterior up to constant).
        scores = []
        for cls in self.classes:
            # Compute Gaussian log-likelihood for each sample.
            ll = multivariate_normal.logpdf(
                x,
                mean=self.mean[cls],
                cov=self.cov[cls],
                allow_singular=True,
            )
            # Add class prior.
            scores.append(ll + self.log_prior[cls])

        # Build score matrix shape: (n_samples, n_classes).
        scores = np.vstack(scores).T
        # Choose class with max score per sample.
        return self.classes[np.argmax(scores, axis=1)]


# Backward-compatible alias.
BayesClassifier = MultivariateGaussianBayesClassifier
