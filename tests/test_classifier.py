import numpy as np
import pytest

from bayes_classifier import (
    BernoulliBayesClassifier,
    CategoricalBayesClassifier,
    GaussianNaiveBayesClassifier,
    MultinomialBayesClassifier,
    MultivariateGaussianBayesClassifier,
    PoissonBayesClassifier,
)


def test_multivariate_gaussian_smoke():
    x = np.array([[1.0, 2.0], [1.2, 1.9], [3.0, 3.1], [2.8, 3.2]])
    y = np.array([0, 0, 1, 1])
    model = MultivariateGaussianBayesClassifier().fit(x, y)
    preds = model.predict(x)
    assert preds.shape == (4,)


def test_multivariate_gaussian_raises_on_feature_mismatch():
    x = np.array([[1.0, 2.0], [1.2, 1.9], [3.0, 3.1], [2.8, 3.2]])
    y = np.array([0, 0, 1, 1])
    model = MultivariateGaussianBayesClassifier().fit(x, y)
    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0, 3.0]]))


def test_gaussian_nb_smoke():
    x = np.array([[1.0, 2.0], [1.2, 1.9], [3.0, 3.1], [2.8, 3.2]])
    y = np.array([0, 0, 1, 1])
    model = GaussianNaiveBayesClassifier().fit(x, y)
    preds = model.predict(x)
    assert preds.shape == (4,)


def test_multinomial_smoke():
    x = np.array([[2, 1], [3, 0], [0, 4], [1, 3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = MultinomialBayesClassifier().fit(x, y)
    preds = model.predict(x)
    assert preds.shape == (4,)


def test_multinomial_rejects_negative_input():
    x = np.array([[2, 1], [3, 0], [0, 4], [1, 3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = MultinomialBayesClassifier().fit(x, y)
    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, -1.0]]))


def test_bernoulli_smoke():
    x = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = BernoulliBayesClassifier().fit(x, y)
    preds = model.predict(x)
    assert preds.shape == (4,)


def test_categorical_smoke():
    x = np.array(
        [
            ["red", "S"],
            ["red", "M"],
            ["blue", "L"],
            ["blue", "M"],
        ],
        dtype=object,
    )
    y = np.array([0, 0, 1, 1])
    model = CategoricalBayesClassifier().fit(x, y)
    preds = model.predict(x)
    assert preds.shape == (4,)


def test_categorical_handles_unseen_value():
    x = np.array(
        [
            ["red", "S"],
            ["red", "M"],
            ["blue", "L"],
            ["blue", "M"],
        ],
        dtype=object,
    )
    y = np.array([0, 0, 1, 1])
    model = CategoricalBayesClassifier().fit(x, y)
    preds = model.predict(np.array([["green", "XL"]], dtype=object))
    assert preds.shape == (1,)


def test_poisson_smoke():
    x = np.array([[1, 0, 2], [2, 1, 1], [7, 8, 6], [6, 7, 9]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = PoissonBayesClassifier().fit(x, y)
    preds = model.predict(x)
    assert preds.shape == (4,)


def test_raises_before_fit():
    x = np.array([[1.0, 2.0]])
    models = [
        MultivariateGaussianBayesClassifier(),
        GaussianNaiveBayesClassifier(),
        MultinomialBayesClassifier(),
        BernoulliBayesClassifier(),
        CategoricalBayesClassifier(),
        PoissonBayesClassifier(),
    ]
    for model in models:
        with pytest.raises(ValueError):
            model.predict(x)
