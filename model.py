"""
model.py
---------

This module defines utilities for persisting and loading scikit‑learn
models used in this proof‑of‑concept. Unlike the previous PyTorch
implementation, we rely on scikit‑learn’s out‑of‑the‑box classifiers to
avoid additional dependencies. Saving and loading models is handled
via ``joblib``, which serialises both the estimator and associated
objects such as the input scaler.
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import joblib


def save_model(model: Any, scaler: Any | None, model_path: str) -> None:
    """Persist a trained scikit‑learn model and optional scaler to disk.

    The objects are packed into a dictionary and serialised using
    ``joblib.dump``. The target file can then be loaded with
    :func:`load_model`.

    Parameters
    ----------
    model : Any
        Trained scikit‑learn estimator.
    scaler : Any or None
        Preprocessing object (e.g. StandardScaler). May be ``None`` if
        no scaling was performed.
    model_path : str
        File path where the dictionary should be saved. The
        directory must already exist.
    """
    state = {"model": model, "scaler": scaler}
    joblib.dump(state, model_path)


def load_model(model_path: str) -> Tuple[Any, Any | None]:
    """Load a scikit‑learn model and scaler from disk.

    Parameters
    ----------
    model_path : str
        Path to the file produced by :func:`save_model`.

    Returns
    -------
    model : Any
        Restored scikit‑learn estimator.
    scaler : Any or None
        Restored scaler used during training, if any.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    state = joblib.load(model_path)
    return state["model"], state.get("scaler")