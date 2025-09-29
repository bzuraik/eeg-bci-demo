"""
data_utils.py
----------------

This module provides utilities for loading and preprocessing EEG data
for the brain‑computer interface proof‑of‑concept. It encapsulates
common operations such as reading the CSV file, normalising the
signals and splitting the data into training and testing subsets.

The dataset used in this project originates from the public EEG Eye
State data set. According to the data description, each row
represents a snapshot of brain activity recorded using a 14‑channel
Emotiv EEG headset. The numeric columns AF3, F7, F3, FC5, T7, P7,
O1, O2, P8, T8, FC6, F4, F8 and AF4 correspond to different sensor
positions, and the final column ``eyeDetection`` indicates whether
the subject's eyes were open (0) or closed (1)【461530564295294†L4-L19】.

To keep this proof‑of‑concept lightweight, preprocessing is limited
to normalising each channel using z‑score scaling. If more
sophisticated filtering (e.g. bandpass filters) is required, this
module can be extended accordingly.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_eeg_data(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    """Load EEG data from a CSV file and return train/test splits.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the EEG measurements. The file
        must have 15 columns: 14 numeric features followed by a binary
        target column named ``eyeDetection``.
    test_size : float, default=0.2
        Fraction of the data to reserve for the test set.
    random_state : int, default=42
        Seed for the pseudo‑random number generator used when splitting.
    normalize : bool, default=True
        If True, the features are z‑score normalised using
        ``sklearn.preprocessing.StandardScaler``.

    Returns
    -------
    X_train : np.ndarray of shape (n_train, n_features)
        Training features.
    X_test : np.ndarray of shape (n_test, n_features)
        Test features.
    y_train : np.ndarray of shape (n_train,)
        Training labels (0 or 1).
    y_test : np.ndarray of shape (n_test,)
        Test labels (0 or 1).
    scaler : StandardScaler or None
        Fitted scaler object if ``normalize`` is True, otherwise ``None``.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    # Load data using pandas for robust CSV parsing
    df = pd.read_csv(csv_path)
    # Ensure the expected columns are present
    expected_features = [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
        "P8", "T8", "FC6", "F4", "F8", "AF4", "eyeDetection",
    ]
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Separate features and labels
    feature_cols = expected_features[:-1]  # all but the last column
    X = df[feature_cols].values.astype(np.float32)
    y = df["eyeDetection"].values.astype(np.int64)

    # Normalize features if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler


def load_test_samples(
    csv_path: str,
    scaler: StandardScaler | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the entire dataset for inference/demo purposes.

    This helper function is used by the Pygame demo to stream EEG
    samples sequentially. If a ``scaler`` is supplied, it will be
    applied to the feature columns; otherwise the raw values are
    returned.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the EEG measurements.
    scaler : StandardScaler or None, default=None
        Scaler object used during training. Providing the same scaler
        ensures that the demo uses the same normalisation as the model
        was trained on.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix. If a scaler is provided the features will be
        normalised accordingly.
    y : np.ndarray of shape (n_samples,)
        Labels corresponding to each sample.
    """
    df = pd.read_csv(csv_path)
    feature_cols = [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
        "P8", "T8", "FC6", "F4", "F8", "AF4",
    ]
    X = df[feature_cols].values.astype(np.float32)
    if scaler is not None:
        X = scaler.transform(X)
    y = df["eyeDetection"].values.astype(np.int64)
    return X, y