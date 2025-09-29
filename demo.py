"""
demo.py
--------

This module runs a simple Tkinter application to showcase how the
trained EEG classifier can be used to control an on‑screen object. The
application streams samples from the dataset, feeds them into the
classifier and maps the predicted classes to horizontal movements of a
square on a canvas:

* Class 0 (eyes open) → move right
* Class 1 (eyes closed) → move left

The demo loops over the dataset continuously. Use the ``--speed``
parameter to control how many samples per second are consumed. Close the
window to exit the demo.
"""

from __future__ import annotations

import argparse
import tkinter as tk
from typing import Tuple

import numpy as np

from data_utils import load_test_samples
from model import load_model


def demo(
    model_path: str,
    data_path: str,
    speed: float = 20.0,
    window_size: Tuple[int, int] = (640, 200),
    square_size: int = 40,
) -> None:
    """Run the Tkinter demo with a trained classifier.

    Parameters
    ----------
    model_path : str
        Path to the saved model (.joblib) produced by ``train.py``.
    data_path : str
        Path to the CSV file containing the EEG data to stream.
    speed : float, default=20.0
        Samples processed per second. Higher values result in faster
        movement.
    window_size : tuple of ints, default=(640, 200)
        Size of the Tkinter window.
    square_size : int, default=40
        Size of the moving square in pixels.
    """
    # Load model and scaler
    model, scaler = load_model(model_path)
    # Load samples (already scaled by scaler inside load_test_samples)
    X_all, y_all = load_test_samples(data_path, scaler)
    num_samples = len(X_all)

    # Tkinter setup
    root = tk.Tk()
    root.title("EEG Brain‑Computer Interface Demo")
    canvas = tk.Canvas(root, width=window_size[0], height=window_size[1], bg="#1e1e1e")
    canvas.pack()

    # Initial square position (centre)
    x_pos = window_size[0] // 2 - square_size // 2
    y_pos = window_size[1] // 2 - square_size // 2
    rect = canvas.create_rectangle(
        x_pos,
        y_pos,
        x_pos + square_size,
        y_pos + square_size,
        fill="green",
    )

    index = 0

    def update():
        nonlocal x_pos, index
        # Predict class for current sample
        sample = X_all[index].reshape(1, -1)
        predicted = int(model.predict(sample)[0])
        # Move square: left or right
        step = 5
        if predicted == 0:
            x_pos += step
            color = "#00c800"  # green for eyes open
        else:
            x_pos -= step
            color = "#c80000"  # red for eyes closed
        # Boundary checks
        x_pos = max(0, min(window_size[0] - square_size, x_pos))
        # Update rectangle position
        canvas.coords(rect, x_pos, y_pos, x_pos + square_size, y_pos + square_size)
        canvas.itemconfig(rect, fill=color)
        # Display status text
        status = f"Sample {index+1}/{num_samples} | Prediction: {predicted}"
        canvas.delete("status")
        canvas.create_text(
            10,
            10,
            anchor="nw",
            text=status,
            fill="white",
            font=("Helvetica", 10),
            tag="status",
        )
        # Increment index and schedule next update
        index = (index + 1) % num_samples
        delay_ms = int(1000 / speed) if speed > 0 else 50
        root.after(delay_ms, update)

    # Start the update loop
    update()
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EEG BCI demo with Tkinter")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.joblib)")
    parser.add_argument("--data", type=str, required=True, help="Path to the CSV dataset for streaming")
    parser.add_argument("--speed", type=float, default=20.0, help="Samples processed per second")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo(args.model, args.data, speed=args.speed)