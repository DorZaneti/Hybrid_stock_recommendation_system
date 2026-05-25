"""
Visualization utilities for stock predictions.

Every plot accepts a `save: bool` and a `show: bool` so the same call works in
both interactive (notebook / desktop) and headless (CI, cron) contexts. PNGs
land under OUTPUT_DIR.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

# Pick a non-interactive backend automatically when no display is available so
# `import utils.visualization` does not crash on a headless box. Honour
# MPLBACKEND when the caller has set it.
if "MPLBACKEND" not in os.environ and not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("./outputs/plots")


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _finalize(fig, filename: Optional[str], save: bool, show: bool) -> None:
    """Save and/or show a figure, then close it to free memory."""
    if save and filename:
        out_dir = _ensure_output_dir()
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_predictions(
    df: pd.DataFrame,
    ticker: str,
    predictions_dict: Dict[str, np.ndarray],
    ticker_index: int,
    num_test_days: int = 5,
    figsize: Tuple[int, int] = (14, 7),
    y_limit_bottom: Optional[float] = None,
    save: bool = True,
    show: bool = False,
) -> None:
    """Plot actual vs predicted prices for a single stock."""
    if ticker not in df.columns:
        logger.warning(f"Ticker {ticker} not found in DataFrame")
        return

    fig, ax = plt.subplots(figsize=figsize)

    actual_prices = df[ticker][-num_test_days:]
    ax.plot(
        df.index[-num_test_days:],
        actual_prices,
        label=f"Actual {ticker} Prices",
        linewidth=2,
        marker="o",
    )

    for model_name, predictions in predictions_dict.items():
        try:
            predicted_values = predictions[-num_test_days:, ticker_index]
            ax.plot(
                df.index[-num_test_days:],
                predicted_values,
                label=f"Predicted {ticker} - {model_name}",
                linestyle="--",
                marker="x",
            )
        except IndexError:
            logger.warning(f"Could not plot predictions for {model_name} - index out of range")

    ax.set_title(f"{ticker} Stock Price Predictions", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if y_limit_bottom is not None:
        ax.set_ylim(bottom=y_limit_bottom)

    fig.tight_layout()
    _finalize(fig, f"predictions_{ticker}.png", save, show)


def plot_multiple_stocks(
    df: pd.DataFrame,
    tickers: List[str],
    predictions_dict: Dict[str, np.ndarray],
    num_test_days: int = 5,
    figsize: Tuple[int, int] = (14, 7),
    y_limit_bottom: Optional[float] = None,
    save: bool = True,
    show: bool = False,
) -> None:
    """Plot predictions for multiple stocks."""
    ticker_list = list(df.columns)

    for ticker in tickers:
        if ticker not in ticker_list:
            logger.warning(f"Ticker {ticker} not in DataFrame columns")
            continue

        ticker_index = ticker_list.index(ticker)
        plot_predictions(
            df,
            ticker,
            predictions_dict,
            ticker_index,
            num_test_days,
            figsize,
            y_limit_bottom,
            save=save,
            show=show,
        )


def plot_training_history(
    losses: List[float],
    title: str = "Training Loss History",
    figsize: Tuple[int, int] = (10, 6),
    filename: str = "training_history.png",
    save: bool = True,
    show: bool = False,
) -> None:
    """Plot training loss over epochs."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(losses, linewidth=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _finalize(fig, filename, save, show)


def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (12, 6),
    save: bool = True,
    show: bool = False,
) -> None:
    """Plot side-by-side bar charts of MSE and RMSE per model."""
    models = list(metrics_dict.keys())
    mse_values = [metrics_dict[m]["MSE"] for m in models]
    rmse_values = [metrics_dict[m]["RMSE"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.bar(models, mse_values, color="steelblue", alpha=0.7)
    ax1.set_title("MSE Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylabel("MSE", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(models, rmse_values, color="coral", alpha=0.7)
    ax2.set_title("RMSE Comparison", fontsize=14, fontweight="bold")
    ax2.set_ylabel("RMSE", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _finalize(fig, "model_comparison.png", save, show)


def plot_directional_accuracy(
    per_model_da: Dict[str, float],
    std_per_model: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save: bool = True,
    show: bool = False,
) -> None:
    """
    Bar chart of directional accuracy (success rate) per model, with 50% baseline.

    Baselines named 'Naive' or 'Majority' are colored differently so the
    LSTM-vs-baseline gap pops visually. If `std_per_model` is provided,
    each bar gets an error bar.
    """
    models = list(per_model_da.keys())
    values = [per_model_da[m] for m in models]
    yerr = [std_per_model.get(m, 0.0) if std_per_model else 0.0 for m in models]
    colors = ["dimgrey" if m in ("Naive", "Majority") else "seagreen" for m in models]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(models, values, color=colors, alpha=0.8, yerr=yerr, capsize=4)
    ax.axhline(50, color="grey", linestyle="--", linewidth=1, label="Random baseline (50%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Directional Accuracy (%)", fontsize=12)
    ax.set_title("Prediction Success Rate by Model (walk-forward mean ± std)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{value:.1f}%",
            ha="center",
            fontsize=10,
        )

    fig.tight_layout()
    _finalize(fig, "directional_accuracy.png", save, show)


def plot_walk_forward_accuracy(
    da_by_model: Dict[str, "pd.DataFrame"],
    figsize: Tuple[int, int] = (12, 6),
    save: bool = True,
    show: bool = False,
) -> None:
    """
    Line chart: x = window start date, y = Directional Accuracy %, one line per model.

    `da_by_model` maps model name to a DataFrame with columns
    `window_start` and `DirectionalAccuracy`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, df in da_by_model.items():
        if df.empty:
            continue
        is_baseline = name in ("Naive", "Majority")
        ax.plot(
            pd.to_datetime(df["window_start"]),
            df["DirectionalAccuracy"],
            marker="o",
            linewidth=2,
            label=name,
            linestyle="--" if is_baseline else "-",
            alpha=0.85,
        )
    ax.axhline(50, color="grey", linestyle=":", linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Window start", fontsize=12)
    ax.set_ylabel("Directional Accuracy (%)", fontsize=12)
    ax.set_title("Walk-forward Success Rate Across Windows", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    _finalize(fig, "walk_forward_accuracy.png", save, show)


def plot_per_stock_metrics(
    metrics_df: pd.DataFrame,
    model_name: str,
    figsize: Tuple[int, int] = (12, 6),
    save: bool = True,
    show: bool = False,
) -> None:
    """
    Per-stock RMSE (left axis, bars) + Directional Accuracy (right axis, line).

    Useful for spotting which tickers the model handles well and which drag the
    aggregate down.
    """
    if metrics_df.empty:
        logger.warning(f"plot_per_stock_metrics({model_name}): empty DataFrame")
        return

    tickers = list(metrics_df.index)
    rmse_values = metrics_df["RMSE"].tolist()
    da_values = metrics_df["DirectionalAccuracy"].tolist()

    fig, ax1 = plt.subplots(figsize=figsize)
    x = np.arange(len(tickers))

    ax1.bar(x, rmse_values, color="steelblue", alpha=0.7, label="RMSE ($)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers, rotation=30, ha="right")
    ax1.set_ylabel("RMSE ($)", fontsize=12, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = ax1.twinx()
    ax2.plot(x, da_values, color="darkorange", marker="o", linewidth=2, label="Directional Accuracy (%)")
    ax2.axhline(50, color="grey", linestyle="--", linewidth=1)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Directional Accuracy (%)", fontsize=12, color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.set_title(f"Per-stock performance — {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    _finalize(fig, f"per_stock_metrics_{safe_name}.png", save, show)


def plot_loss_curves(
    losses_per_model: Dict[str, object],
    figsize: Tuple[int, int] = (12, 6),
    save: bool = True,
    show: bool = False,
) -> None:
    """
    Overlay train (solid) + val (dashed) loss curves per model on one log-y chart.

    Accepts either:
      - the legacy shape `{model_name: [losses]}` (val curve omitted), or
      - the new shape `{model_name: {"train_losses": [...], "val_losses": [...]}}`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")

    for i, (model_name, payload) in enumerate(losses_per_model.items()):
        if isinstance(payload, dict):
            train_losses = payload.get("train_losses", [])
            val_losses = payload.get("val_losses", [])
        else:
            train_losses, val_losses = list(payload), []

        color = cmap(i % 10)
        if train_losses:
            ax.plot(
                range(1, len(train_losses) + 1),
                train_losses,
                linewidth=2,
                label=f"{model_name} (train)",
                color=color,
            )
        if val_losses:
            ax.plot(
                range(1, len(val_losses) + 1),
                val_losses,
                linewidth=2,
                linestyle="--",
                label=f"{model_name} (val)",
                color=color,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE, log scale)", fontsize=12)
    ax.set_title("Training Loss Curves (train solid, val dashed)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", ncol=2, fontsize=9)
    fig.tight_layout()
    _finalize(fig, "loss_curves.png", save, show)


def save_plot(
    filename: str,
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> None:
    """Save the current pyplot figure to disk."""
    try:
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Plot saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
        raise
