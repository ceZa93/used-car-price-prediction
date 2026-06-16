"""
Shared utilities for the used-car-price-prediction pipeline.

Consolidates repeated patterns: directory constants, artifact I/O,
section/step logging, regression metrics, and plot saving.
"""
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── constants ────────────────────────────────────────────────────────────────
REFERENCE_YEAR = 2024
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")


# ── logging helpers ──────────────────────────────────────────────────────────
def log_section(title: str) -> None:
    print(f"\n=== {title} ===")


def log_step(message: str) -> None:
    print(f"\u2713 {message}")


# ── artifact I/O ─────────────────────────────────────────────────────────────
def save_artifact(obj: object, filename: str) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(obj, path)
    log_step(f"Saved: {path}")
    return path


def load_artifact(filename: str) -> object:
    return joblib.load(MODELS_DIR / filename)


# ── metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ── plot helpers ─────────────────────────────────────────────────────────────
def save_plot(filename: str) -> Path:
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log_step(f"Grafik sačuvan: {path}")
    return path
