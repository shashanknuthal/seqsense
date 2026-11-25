"""Small Flask interface for predictions from uploaded 128 x 9 sensor windows."""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import torch
from flask import Flask, flash, redirect, render_template, request, url_for

from models import build_model

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
app = Flask(__name__)
app.secret_key = os.environ.get("SEQSENSE_SECRET_KEY", "local-development-only-change-before-deployment")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]


def available_models():
    return sorted(p.stem for p in ARTIFACTS.glob("*.pt"))


def load_window(file_storage):
    raw = file_storage.read().decode("utf-8-sig")
    # genfromtxt tolerates a single optional text header by returning an all-NaN row.
    values = np.genfromtxt(io.StringIO(raw), delimiter=",", dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    values = values[~np.isnan(values).all(axis=1)]
    if values.shape != (128, 9) or not np.isfinite(values).all():
        raise ValueError("Upload a CSV containing exactly 128 rows and 9 numeric columns, in the documented channel order.")
    return values


@app.get("/")
def index():
    return render_template("index.html", models=available_models(), channels=CHANNELS, prediction=None)


@app.post("/predict")
def predict():
    models = available_models()
    model_name = request.form.get("model", "")
    upload = request.files.get("window")
    if model_name not in models:
        flash("Train a model first, then select one of the available checkpoints.")
        return redirect(url_for("index"))
    if not upload or not upload.filename:
        flash("Choose a sensor window CSV file.")
        return redirect(url_for("index"))
    try:
        window = load_window(upload)
        checkpoint = torch.load(ARTIFACTS / f"{model_name}.pt", map_location="cpu", weights_only=False)
        model = build_model(checkpoint["model_name"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
        normalized = (window - mean) / np.maximum(std, 1e-6)
        with torch.no_grad():
            probabilities = torch.softmax(model(torch.from_numpy(normalized[None, ...])), dim=1)[0].numpy()
        result = {"label": LABELS[int(probabilities.argmax())],
                  "scores": sorted(zip(LABELS, probabilities.tolist()), key=lambda item: item[1], reverse=True)}
        return render_template("index.html", models=models, channels=CHANNELS, prediction=result)
    except (ValueError, UnicodeDecodeError, OSError, KeyError, RuntimeError) as exc:
        flash(str(exc))
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
