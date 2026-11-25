# SeqSense: Sequence Classification for Human Activity Recognition

SeqSense is a Git-ready starter project for classifying human activity from smartphone inertial sensor windows. It implements a 1-D CNN, vanilla RNN, LSTM, GRU, and CNN-LSTM comparison, a training/evaluation pipeline, and a Flask interface for CSV window predictions.

> **Research prototype:** This is not a health, safety, or clinical product. The model scores depend on the UCI HAR dataset and may not transfer to different devices, sensor placements, or populations.
>
> **Results provenance:** The benchmark values and experiment findings below are transcribed from the project brief and have not been independently reproduced against original checkpoints or metric files. Use the provided scripts to reproduce results before treating them as verified.

## Features

- UCI HAR preprocessing from the nine raw inertial-signal channels
- Subject-independent train/test split preserved from the source dataset
- Train-set-only per-channel normalization
- Five PyTorch architectures: 1-D CNN, RNN, LSTM, GRU, CNN-LSTM
- Early stopping, learning-rate scheduling, gradient clipping, fixed seeds
- Accuracy, macro-F1, class-wise metrics, confusion matrix, parameter count, and CPU latency output
- Flask upload interface for a single 128 x 9 sensor window

## Dataset

This project uses the [UCI Human Activity Recognition Using Smartphones dataset](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones), which the UCI record lists under CC BY 4.0. The dataset paper is Anguita et al., [“A Public Domain Dataset for Human Activity Recognition Using Smartphones”](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf), ESANN 2013.

The dataset includes 30 volunteers, six activity labels, 128-sample windows at 50 Hz, and nine channels. The original split contains 7,352 training windows from 21 subjects and 2,947 test windows from nine held-out subjects. We use `body_acc_x/y/z`, `body_gyro_x/y/z`, and `total_acc_x/y/z`.

Download and extract the dataset from UCI. The raw and prepared dataset files are not included or committed by default. If using a mirror, record its URL and verify the redistribution terms. Cite the dataset and paper in any report or derivative release.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download and extract the UCI HAR dataset, then point the preprocessing command at its `UCI HAR Dataset` directory:

```bash
python prepare_data.py --dataset-root "/path/to/UCI HAR Dataset"
```

The resulting `data/har.npz` is ignored by Git. Train a model with:

```bash
python train.py --model cnn
```

Choose `cnn`, `rnn`, `lstm`, `gru`, or `cnn_lstm`. Training writes a checkpoint and metrics JSON to `artifacts/`. To train each architecture:

```bash
for model in cnn rnn lstm gru cnn_lstm; do python train.py --model "$model"; done
```

On Windows PowerShell:

```powershell
'cnn','rnn','lstm','gru','cnn_lstm' | ForEach-Object { python train.py --model $_ }
```

Launch the local Flask interface after training:

```bash
python app.py
```

Open <http://127.0.0.1:5000>. Upload a CSV containing exactly 128 rows and 9 numeric columns in this order: `body_acc_x`, `body_acc_y`, `body_acc_z`, `body_gyro_x`, `body_gyro_y`, `body_gyro_z`, `total_acc_x`, `total_acc_y`, `total_acc_z`.

## Model comparison reported in the project brief

| Model | Accuracy | Macro-F1 | Parameters | Train time | Epochs | CPU latency/window |
|---|---:|---:|---:|---:|---:|---:|
| 1-D CNN | 92.37% | 0.925 | 166,086 | 34.5 s | 10 | 0.40 ms |
| CNN-LSTM | 91.38% | 0.915 | 57,414 | 33.4 s | 18 | 0.72 ms |
| RNN | 90.46% | 0.905 | 13,510 | 150.6 s | 30 | 1.07 ms |
| LSTM | 90.13% | 0.902 | 52,870 | 52.2 s | 17 | 1.82 ms |
| GRU | 89.62% | 0.897 | 39,750 | 131.2 s | 17 | 3.67 ms |

The brief reports that the CNN performed best on this benchmark. Its main error was confusion between `SITTING` and `STANDING`. The scores above are **reported, not independently verified**. They may differ from this scaffold because of implementation details, preprocessing, hardware timing, and seed behavior. Re-run the training and update the table from generated metric files before publishing results as reproduced.

## Sequence-dependency experiments

The brief also reports two experiments:

1. **Context length on UCI HAR:** compare windows cropped to 16, 32, 64, and 128 samples. The reported recurrent results show LSTM/GRU improving as more context is available; the vanilla RNN peaked at 64 in this run.
2. **Synthetic long-term memory:** a sequence begins with a class cue followed by random distractors. Reported LSTM and GRU results retained the cue at longer sequence lengths, while the vanilla RNN approached chance at 100–200 steps. A global-pooling CNN can detect the cue anywhere; that result is not evidence of recurrent memory.

The detailed experiment driver and original `experiments.json` were not included in the brief, so these experiments are described but not implemented in this starter. Add the original script or request implementation before claiming these results are reproducible from this repository.

## Web app scope

The supplied brief mentions live and uploaded-file predictions. This repository implements the **uploaded CSV window** path. Browser sensor streaming is not included: the UCI channels include processed body/total acceleration, so raw phone sensor events need a matching preprocessing pipeline before they can be fed to these models. Do not label the current app as real-time phone inference until that path exists and has been validated.

## Repository layout

```text
seqsense/
├── app.py                 # Flask CSV prediction interface
├── models.py              # Five sequence classifiers
├── prepare_data.py        # UCI inertial-signal data preparation
├── train.py                # Training, metrics, and checkpoint export
├── requirements.txt
├── templates/index.html
├── data_README.md          # Dataset handling notes
├── data/                   # Generated locally; excluded from Git
└── artifacts/              # Checkpoints and metric outputs, generated locally
```

## Data and privacy

The UCI data is a public research dataset, but keep participant-linked or personally collected sensor data out of the repository. Do not upload private recordings, API keys, or environment files. Review the UCI license and attribution requirements before redistributing any dataset files.

## Citation

Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN 2013. [Paper](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf) · [UCI dataset record](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones).

## Before publishing

- Run the training pipeline on a clean environment and verify metrics.
- Add screenshots of the app and plots generated from the actual run.
- Include the official data source and a clear dataset preparation record.
- Add the original dependency lock/config if available.
- Choose a repository license only if all contributors agree to license the project publicly.
- Confirm public attribution preferences with teammates; student IDs and guide placeholders are intentionally omitted.
