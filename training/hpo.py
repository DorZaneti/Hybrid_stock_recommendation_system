"""
Optuna hyperparameter optimisation for the multi-feature StockLSTM (Model 0 variant).

Runs the data pipeline ONCE, then searches over:
    hidden_size, num_layers, dropout, learning_rate, direction_loss_weight

Objective: validation-set directional accuracy (maximise).
Pruner:    MedianPruner — kills trials whose val-MSE trajectory falls below the
           per-step median after a short warm-up.
Sampler:   TPE (default in Optuna) — Bayesian inference over the search space.

Usage
-----
    python training/hpo.py                     # 50 trials, default study name
    python training/hpo.py --n_trials 100
    python training/hpo.py --n_trials 30 --study_name quick_sweep

Outputs
-------
    outputs/best_hpo.yaml   — best params + achieved val DA
    outputs/hpo_history.csv — per-trial params + val DA (for analysis)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import yaml
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Make sure project root is on sys.path when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config_loader import load_config, get_config_value
from utils.logger import get_logger
from utils.device import get_device, set_random_seed
from utils.metrics import directional_accuracy

from data.download import download_stock_data, download_macro_data
from data.features import calculate_all_features, calculate_macro_features
from data.similarity import find_similar_stocks
from data.returns import compute_forward_returns, compute_past_returns
from data.preprocessing import (
    prepare_stock_dataframe,
    prepare_multifeature_array,
    create_sequences_xy,
    fit_scaler_on_train,
    split_train_val,
    FEATURES_PER_STOCK,
    MACRO_FEATURES,
)

from models.lstm import StockLSTM
from training.trainer import LSTMTrainer
from training.walk_forward import LSTMPredictor

logger = get_logger(__name__)

OUTPUTS_DIR = Path("./outputs")


# ---------------------------------------------------------------------------
# Data pipeline (runs once; results are reused across all trials)
# ---------------------------------------------------------------------------

def build_dataset(config: Dict[str, Any], device: torch.device, seq_length_override: int = None):
    """
    Run the full data pipeline and return all arrays needed by the objective.
    Mirrors the logic in main.py but returns tensors instead of training models.

    seq_length_override: if provided, use this instead of config model.sequence_length.
    This lets the Optuna trial vary seq_length without rebuilding the full dataset —
    the underlying feature/target arrays are built at the max seq_length, and each
    trial slices sequences at its own length from those same arrays.
    """
    tickers = get_config_value(config, "data", "tickers")
    start_date = get_config_value(config, "data", "start_date")
    end_date = get_config_value(config, "data", "end_date")
    train_end_date = get_config_value(config, "data", "train_end_date")
    val_days = get_config_value(config, "training", "validation_days", default=120)
    seq_length = seq_length_override or get_config_value(config, "model", "sequence_length")
    horizon = int(get_config_value(config, "target", "horizon_days", default=1))
    target_type = get_config_value(config, "target", "type", default="log_returns")
    direction_mode = (target_type == "direction")

    logger.info("[HPO] Downloading stock data…")
    data = download_stock_data(tickers, start_date, end_date)
    feature_config = get_config_value(config, "features")
    for ticker in data:
        data[ticker] = calculate_all_features(
            data[ticker],
            rsi_window=feature_config["rsi"]["window"],
            momentum_window=feature_config["momentum"]["window"],
            ma_window=feature_config["moving_average"]["window"],
            bb_window=feature_config["bollinger_bands"]["window"],
            bb_std=feature_config["bollinger_bands"]["std_multiplier"],
            macd_fast=feature_config.get("macd", {}).get("fast", 12),
            macd_slow=feature_config.get("macd", {}).get("slow", 26),
            macd_signal=feature_config.get("macd", {}).get("signal", 9),
            atr_window=feature_config.get("atr", {}).get("window", 14),
            volume_window=feature_config.get("log_volume", {}).get("window", 20),
        )

    sim_cfg = get_config_value(config, "similarity")
    similar_stocks = find_similar_stocks(
        data, sim_cfg["base_ticker"], sim_cfg["num_similar_stocks"],
        train_end_date=train_end_date,
    )
    logger.info(f"[HPO] Similar stocks: {similar_stocks}")

    macro_cfg = get_config_value(config, "macro", default={}) or {}
    macro_df = None
    if macro_cfg.get("enabled", False):
        macro_raw = download_macro_data(
            macro_cfg.get("tickers", ["^VIX", "^GSPC"]), start_date, end_date
        )
        if macro_raw:
            try:
                macro_df = calculate_macro_features(
                    macro_raw, vix_change_window=macro_cfg.get("vix_change_window", 5)
                )
            except Exception as e:
                logger.warning(f"[HPO] Macro feature error: {e}; continuing without")

    prices_df = prepare_stock_dataframe(data, similar_stocks, start_date, end_date).sort_index()
    features_df, _, _ = prepare_multifeature_array(
        data, similar_stocks, start_date, end_date, macro_df=macro_df
    )
    forward_returns = compute_forward_returns(prices_df, horizon=horizon)
    past_returns_df = compute_past_returns(prices_df, horizon=horizon)

    common_idx = (
        features_df.index
        .intersection(forward_returns.index)
        .intersection(past_returns_df.index)
    )
    features_df = features_df.reindex(common_idx)
    targets_df = forward_returns.reindex(common_idx)[similar_stocks]
    prices_df = prices_df.reindex(common_idx)

    n_features = len(FEATURES_PER_STOCK)
    n_macro = max(0, features_df.shape[1] - len(similar_stocks) * n_features)

    train_end_ts = pd.Timestamp(train_end_date)
    train_full_mask = features_df.index < train_end_ts
    n_train_full = int(train_full_mask.sum())

    n_train, n_val = split_train_val(n_train_full, val_days)

    features_arr = features_df.values
    targets_arr = targets_df.values

    scaled_features, input_scaler = fit_scaler_on_train(features_arr, n_train, "standard")

    if direction_mode:
        # Binary targets: 1 = up, 0 = down. No output scaler needed.
        scaled_targets = (targets_arr > 0).astype(np.float32)
        output_scaler = None
    else:
        scaled_targets, output_scaler = fit_scaler_on_train(targets_arr, n_train, "standard")

    X_all, y_all = create_sequences_xy(scaled_features, scaled_targets, seq_length)
    target_dates = features_df.index[seq_length:]

    train_mask = target_dates < features_df.index[n_train]
    val_mask = (target_dates >= features_df.index[n_train]) & (
        target_dates < features_df.index[n_train_full]
    )

    X_train = torch.tensor(X_all[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y_all[train_mask], dtype=torch.float32)
    X_val_t = torch.tensor(X_all[val_mask], dtype=torch.float32)
    y_val_t = torch.tensor(y_all[val_mask], dtype=torch.float32)

    val_dates = target_dates[val_mask]
    val_anchor_prices = prices_df.shift(1).loc[val_dates, similar_stocks].values
    X_val_unscaled = (X_all[val_mask] * input_scaler.scale_) + input_scaler.mean_
    y_val_returns = targets_arr[seq_length:][val_mask]   # always raw returns (for metrics)

    logger.info(
        f"[HPO] Dataset ready: train={len(X_train)}, val={len(X_val_t)} | "
        f"input_dim={X_train.shape[-1]}, output_dim={y_train.shape[-1]} | "
        f"n_features={n_features}, n_macro={n_macro} | "
        f"mode={'direction' if direction_mode else 'log_returns'}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_val_unscaled": X_val_unscaled,
        "val_anchor_prices": val_anchor_prices,
        "y_val_returns": y_val_returns,
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "num_stocks": len(similar_stocks),
        "n_features": n_features,
        "n_macro": n_macro,
        "device": device,
        "direction_mode": direction_mode,
        "features_arr": features_arr,   # kept for seq_length re-slicing per trial
        "scaled_targets_arr": scaled_targets,
        "target_dates_full": features_df.index,
        "n_train": n_train,
        "n_train_full": n_train_full,
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(ds: Dict[str, Any]):
    """Return an Optuna objective function closed over the pre-built dataset.

    Search space (direction mode):
        hidden_size, num_layers, dropout, lr, seq_length
        direction_loss_weight is EXCLUDED — in direction_mode it is forced to 0
        (the BCE is the entire loss, not an auxiliary term).

    Search space (log_returns mode):
        same as above, plus direction_loss_weight as an auxiliary BCE weight.

    Objective: minimise best val loss (BCE in direction mode, MSE otherwise).
    BCE rewards genuine per-stock probability calibration, not just majority-
    class prediction, making it more regime-fair than val directional accuracy.
    """
    input_scaler = ds["input_scaler"]
    num_stocks = ds["num_stocks"]
    n_features = ds["n_features"]
    n_macro = ds["n_macro"]
    device = ds["device"]
    direction_mode = ds.get("direction_mode", False)
    multi_input_size = num_stocks * n_features + n_macro

    # Arrays for seq_length re-slicing (each trial may use a different length).
    features_arr_scaled = (ds["features_arr"] - input_scaler.mean_) / input_scaler.scale_
    scaled_targets_arr  = ds["scaled_targets_arr"]
    target_dates_full   = ds["target_dates_full"]
    n_train             = ds["n_train"]
    n_train_full        = ds["n_train_full"]
    val_anchor_prices   = ds["val_anchor_prices"]   # always aligned to default seq_length val window
    y_val_returns       = ds["y_val_returns"]

    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_categorical("hidden_size", [64, 100, 128, 256])
        num_layers  = trial.suggest_int("num_layers", 1, 4)
        dropout     = trial.suggest_float("dropout", 0.05, 0.50)
        lr          = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        seq_length  = trial.suggest_categorical("seq_length", [20, 30, 40, 60])

        # direction_loss_weight only makes sense in log_returns mode.
        direction_loss_weight = (
            0.0 if direction_mode
            else trial.suggest_float("direction_loss_weight", 0.0, 0.5)
        )

        # Re-slice sequences at this trial's seq_length.
        X_all_t, y_all_t = create_sequences_xy(
            features_arr_scaled, scaled_targets_arr, seq_length
        )
        t_dates = target_dates_full[seq_length:]
        train_mask_t = t_dates < target_dates_full[n_train]
        val_mask_t   = (t_dates >= target_dates_full[n_train]) & (
            t_dates < target_dates_full[n_train_full]
        )
        X_tr = torch.tensor(X_all_t[train_mask_t], dtype=torch.float32)
        y_tr = torch.tensor(y_all_t[train_mask_t], dtype=torch.float32)
        X_vl = torch.tensor(X_all_t[val_mask_t],   dtype=torch.float32)
        y_vl = torch.tensor(y_all_t[val_mask_t],   dtype=torch.float32)

        model = StockLSTM(
            input_size=multi_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=num_stocks,
            dropout=dropout,
        ).to(device)

        trainer = LSTMTrainer(
            model, device,
            learning_rate=lr,
            direction_loss_weight=direction_loss_weight,
        )
        trainer.direction_mode = direction_mode

        # Train with aggressive early stopping to keep each trial fast.
        result = trainer.train(
            X_tr, y_tr,
            num_epochs=100,
            log_interval=9999,          # silence per-epoch logs during HPO
            X_val=X_vl, y_val=y_vl,
            early_stopping_patience=7,
            early_stopping_enabled=True,
        )

        # Report per-epoch val loss for MedianPruner.
        for step, vl in enumerate(result["val_losses"]):
            trial.report(vl, step=step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Objective: minimise best val loss.
        # In direction mode: BCE loss — rewards calibrated per-stock probabilities.
        # In log_returns mode: MSE — regime-neutral magnitude accuracy.
        # Both are computed on the raw logit/return outputs of the trainer,
        # making them more informative than price-level RMSE.
        best_val = result["best_val"]
        if best_val is None:
            best_val = result["val_losses"][-1] if result["val_losses"] else float("inf")

        return float(best_val)

    return objective


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for StockLSTM")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--study_name", type=str, default="stock_lstm_hpo",
                        help="Optuna study name")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Stop after this many seconds (optional)")
    args = parser.parse_args()

    config = load_config()
    device = get_device(get_config_value(config, "training", "device_priority"))
    set_random_seed(get_config_value(config, "training", "random_seed", default=42), device)

    target_type = get_config_value(config, "target", "type", default="log_returns")
    direction_mode = (target_type == "direction")
    objective_label = "val BCE" if direction_mode else "val MSE"

    logger.info(
        f"[HPO] Starting Optuna study '{args.study_name}' | {args.n_trials} trials | "
        f"device={device} | mode={target_type} | objective=minimize {objective_label}"
    )

    # Build dataset once (uses max seq_length from config; trials re-slice as needed).
    ds = build_dataset(config, device)

    # Create and run the study.
    study = optuna.create_study(
        direction="minimize",   # minimise val BCE (direction mode) or val MSE (log_returns)
        study_name=args.study_name,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=15),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        make_objective(ds),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # ── Results ──────────────────────────────────────────────────────────────
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best {objective_label}: {best.value:.6f}")
    print(f"Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save best params to YAML for easy copy-paste into config.yaml.
    best_path = OUTPUTS_DIR / "best_hpo.yaml"
    with open(best_path, "w") as f:
        yaml.dump(
            {
                f"best_{objective_label.replace(' ', '_')}": float(best.value),
                "target_type": target_type,
                "params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v)
                           for k, v in best.params.items()},
            },
            f,
            default_flow_style=False,
        )
    logger.info(f"[HPO] Best params saved to {best_path}")

    # Save per-trial history.
    hist_rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = {"trial": t.number, objective_label.replace(" ", "_"): t.value}
            row.update(t.params)
            hist_rows.append(row)
    if hist_rows:
        sort_col = objective_label.replace(" ", "_")
        hist_df = pd.DataFrame(hist_rows).sort_values(sort_col, ascending=True)  # lower is better
        hist_path = OUTPUTS_DIR / "hpo_history.csv"
        hist_df.to_csv(hist_path, index=False)
        logger.info(f"[HPO] Trial history saved to {hist_path}")
        print(f"\nTop-5 trials (lowest {objective_label}):\n{hist_df.head(5).to_string(index=False)}")


if __name__ == "__main__":
    main()
