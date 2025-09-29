# EEG Brain‑Computer Interface Proof‑of‑Concept

This repository demonstrates a complete workflow for translating
electroencephalography (EEG) brain signals into digital commands that
control a simple on‑screen object. The project is designed to be
self‑contained and reproducible: it downloads open data, trains a
classifier and shows the model’s predictions driving a graphical
interface.

## Overview

Brain–computer interfaces (BCIs) offer a way to interact with
computers directly through neural activity. While sophisticated
systems such as Neuralink decode complex motor intentions, even
consumer‑grade EEG devices can detect simple states like whether a
person’s eyes are open or closed. In this project we:

1. **Select a publicly available EEG dataset.**  We use the
   *EEG Eye State* data set where 14 electrodes on an Emotiv
   headset recorded 117 seconds of brain activity while a subject’s
   eyes alternated between open and closed. The final column
   `eyeDetection` contains the ground truth labels (`0` open,
   `1` closed)【461530564295294†L4-L19】.  The full file contains 14 980
   samples with no missing values【300254446879847†L45-L50】.
2. **Preprocess and split the data.**  Signals are z‑score normalised
   channel‑wise and divided into training and test sets.
3. **Train a classifier.**  A simple multi‑layer perceptron (MLP)
   implemented using scikit‑learn learns to distinguish open vs.
   closed eyes. With a single hidden layer of 64 neurons the
   classifier achieves around 80 % test accuracy.
4. **Map model outputs to actions.**  In the demo application each
   prediction moves a coloured square: class 0 (eyes open) makes the
   square move right and turn green, whereas class 1 (eyes closed)
   moves it left and turns it red.

## Getting started

1. **Clone or download the repository.**  All source code and the
   pre‑packaged dataset (`data/eeg_eye_state_full.csv`) live under
   `brain_bc_project/`.
2. **Install dependencies.**  The code relies only on built‑in
   Python libraries plus `pandas`, `scikit‑learn` and
   `joblib`. Install them via pip:

   ```bash
   pip install pandas scikit-learn joblib
   ```
3. **Train the model.**  Run the training script to fit the MLP and
   save the trained model and scaler:

   ```bash
   python src/train.py --data data/eeg_eye_state_full.csv --output models/eeg_classifier.joblib --epochs 200
   ```
   After training you should see training and test accuracy printed
   in the console. The model (together with the normalisation scaler) is
   stored in `models/eeg_classifier.joblib`.
4. **Run the demo.**  Launch the graphical interface to visualise
   the classifier’s output in real time:

   ```bash
   python src/demo.py --model models/eeg_classifier.joblib --data data/eeg_eye_state_full.csv --speed 20
   ```
   A window will open showing a coloured square moving horizontally.
   Each incoming EEG sample is passed through the model; if the
   prediction is **0** (eyes open) the square moves right and turns
   green, otherwise it moves left and turns red. Sample number and
   predicted class are displayed at the top left.

## Interface snapshot

Below is a mock‑up of the demo window. A dark canvas contains a small
square that changes colour and position in response to the classifier’s
predictions. The text overlay shows the current sample index and
predicted class.

![Brain–computer interface demo]({{file:file-CQq1T32bUVYkLHmTTZ7qFP}})

## Real‑world implications

This proof‑of‑concept may be simple but it illustrates the key
components of BCIs: capturing neural activity, translating signals
into states and using those states to drive an external device. The
same pipeline could be extended to more complex tasks such as motor
imagery (left/right hand movement, foot movement etc.) using data from
the PhysioNet *EEG Motor Movement/Imagery* dataset【227508551410058†L78-L137】. With a richer
mapping and a more capable classifier you could control a wheelchair,
play a video game or interact with a virtual environment without any
physical input.

## Contributing

Contributions are welcome! Feel free to submit pull requests for
improvements such as additional datasets, alternative models or more
interactive demos. Please ensure that any added data is properly
licensed and cited.