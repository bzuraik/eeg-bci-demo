"""
train.py
---------

Train a scikit‑learn classifier on the EEG Eye State dataset. The
script normalises the features, splits the data into training and test
subsets, fits a multi‑layer perceptron (MLP) classifier and reports
accuracy metrics. The trained model and scaler are saved to disk for
later use by the demo application.

Example usage:

```
python src/train.py --data data/eeg_eye_state_full.csv --output models/eeg_classifier.joblib --epochs 20
```

Note that scikit‑learn’s MLPClassifier uses the concept of
“iterations” rather than epochs. The ``--epochs`` argument is mapped
to the ``max_iter`` parameter of the classifier.
"""

from __future__ import annotations

import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from data_utils import load_eeg_data
from model import save_model


def train_model(
    data_path: str,
    output_path: str,
    epochs: int = 20,
    hidden_size: int = 64,
    learning_rate: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Train an MLP classifier on the EEG dataset and save it.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the dataset.
    output_path : str
        Path where the trained model will be saved (e.g. .joblib).
    epochs : int, default=20
        Maximum number of training iterations for the MLP classifier.
    hidden_size : int, default=64
        Size of the hidden layer.
    learning_rate : float, default=1e-3
        Learning rate for the stochastic gradient solver.
    test_size : float, default=0.2
        Fraction of data reserved for testing.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_eeg_data(
        data_path,
        test_size=test_size,
        random_state=random_state,
        normalize=True,
    )

    # Define MLP classifier
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_size,),
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate,
        max_iter=epochs,
        random_state=random_state,
    )
    # Fit classifier
    clf.fit(X_train, y_train)

    # Evaluate on training and test sets
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy:     {test_acc:.4f}")

    # Save model and scaler
    save_model(clf, scaler, output_path)
    print(f"Model saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG eye state classifier using scikit‑learn")
    parser.add_argument("--data", type=str, required=True, help="Path to the EEG CSV dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save the trained model (.joblib)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training iterations for the MLP")
    parser.add_argument("--hidden_size", type=int, default=64, help="Number of hidden units in the MLP")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for Adam solver")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data used for testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        raise FileNotFoundError(f"Output directory does not exist: {out_dir}")
    train_model(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        random_state=42,
    )