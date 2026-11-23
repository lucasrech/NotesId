"""Classificadores clássicos (sem machine learning) para escolha de corda e casa."""

from guitar_string_identifier.classifier.classical import (
    DEFAULT_PRIOR_B0,
    GaussianStringClassifier,
)

__all__ = ["DEFAULT_PRIOR_B0", "GaussianStringClassifier"]
