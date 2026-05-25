"""

End-to-end pipeline:
  1. Download prices + compute technical indicators + pick similar stocks.
  2. Convert prices -> log-returns (stationary, sign == direction).
  3. Time-ordered split into train / val / out-of-sample.
  4. Fit StandardScaler on TRAIN ONLY, then transform everything.
  5. Train four LSTM variants with early stopping on val MSE.
  6. Walk-forward evaluate every model (and two baselines) across many windows
     of the OOS slice; aggregate RMSE / MAPE / DirectionalAccuracy mean ± std.
  7. Save CSVs + PNGs for the Streamlit dashboard.

Outputs:
  - saved_models/model_*.pt
  - outputs/metrics.csv                aggregate per-model (mean / std / min / max)
  - outputs/walk_forward_results.csv   one row per (model, window)
  - outputs/predictions.csv            long-format predictions (pooled OOS)
  - outputs/losses.csv                 train & val loss per epoch
  - outputs/per_stock/<model>.csv      per-stock RMSE/MAPE/DirectionalAccuracy
  - outputs/plots/*.png
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from utils.config_loader import load_config, get_config_value
from utils.logger import setup_logger
from utils.device import get_device, set_random_seed
from utils.visualization import (
    plot_multiple_stocks,
    plot_model_comparison,
    plot_directional_accuracy,
    plot_per_stock_metrics,
    plot_loss_curves,
    plot_walk_forward_accuracy,
)

from data.download import download_stock_data, download_macro_data
from data.features import calculate_all_features, calculate_macro_features
from data.similarity import find_similar_stocks
from data.returns import (
    prices_to_log_returns,
    compute_forward_returns,
    compute_past_returns,
)
from data.preprocessing import (
    prepare_stock_dataframe,
    prepare_multifeature_array,
    create_sequences,
    create_sequences_xy,
    fit_scaler_on_train,
    split_train_val,
    FEATURES_PER_STOCK,
    MACRO_FEATURES,
)

from models.lstm import StockLSTM, SingleFeatureLSTM
from models.baselines import NaivePersistence, MajorityClass
from models.ensemble import EnsembleAverage, EnsembleWeighted
from models.persistence import save_model
from models.xgb_model import XGBoostPredictor, HAS_XGB
from utils.metrics import directional_accuracy
from training.trainer import LSTMTrainer, SingleFeatureTrainer
from training.walk_forward import (
    LSTMPredictor,
    SingleFeaturePredictor,
    walk_forward_predict,
    windows_to_frame,
    pool_windows,
    pooled_per_stock,
)


OUTPUTS_DIR = Path("./outputs")


def main():
    print("=" * 70)
    print("STOCK PRICE PREDICTION SYSTEM")
    print("=" * 70)

    config = load_config()
    logger = setup_logger(
        __name__,
        level=get_config_value(config, "logging", "level", default="INFO"),
        log_file=get_config_value(config, "logging", "file"),
        console_output=get_config_value(config, "logging", "console_output", default=True),
    )
    logger.info("Starting stock price prediction system (log-return target)")

    device = get_device(get_config_value(config, "training", "device_priority"))
    set_random_seed(get_config_value(config, "training", "random_seed", default=42), device)

    # ---------- Data ----------
    logger.info("Phase 1: Downloading stock data")
    tickers = get_config_value(config, "data", "tickers")
    start_date = get_config_value(config, "data", "start_date")
    end_date = get_config_value(config, "data", "end_date")
    data = download_stock_data(tickers, start_date, end_date)
    logger.info(f"Downloaded data for {len(data)} stocks")

    logger.info("Phase 2: Calculating technical indicators (for similarity)")
    feature_config = get_config_value(config, "features")
    for ticker in data.keys():
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

    logger.info("Phase 3: Finding similar stocks")
    sim_config = get_config_value(config, "similarity")
    base_ticker = sim_config["base_ticker"]
    num_similar = sim_config["num_similar_stocks"]
    train_end_date = get_config_value(config, "data", "train_end_date")
    similar_stocks = find_similar_stocks(
        data, base_ticker, num_similar, train_end_date=train_end_date
    )
    logger.info(f"Similar stocks to {base_ticker}: {similar_stocks}")

    # ---------- Macro / regime features (VIX + S&P 500) ----------
    macro_cfg = get_config_value(config, "macro", default={})
    macro_df = None
    macro_enabled = macro_cfg.get("enabled", False) if macro_cfg else False
    if macro_enabled:
        macro_tickers = macro_cfg.get("tickers", ["^VIX", "^GSPC"])
        vix_change_window = macro_cfg.get("vix_change_window", 5)
        logger.info(f"Phase 3b: Downloading macro data: {macro_tickers}")
        macro_raw = download_macro_data(macro_tickers, start_date, end_date)
        if macro_raw:
            try:
                macro_df = calculate_macro_features(
                    macro_raw,
                    vix_change_window=vix_change_window,
                )
                logger.info(
                    f"Macro features ready: {list(macro_df.columns)} "
                    f"({len(macro_df)} rows)"
                )
            except Exception as exc:
                logger.warning(f"Macro feature computation failed: {exc}; continuing without")
                macro_df = None
        else:
            logger.warning("Macro download returned no data; continuing without macro features")

    # ---------- Prices + multi-feature inputs ----------
    # Prices frame still needed for anchors / dashboard plotting.
    prices_df = prepare_stock_dataframe(data, similar_stocks, start_date, end_date).sort_index()
    logger.info(f"Prices frame: {prices_df.shape}")

    # Build per-stock feature blocks (Phase 2). Features stay at 1-day resolution
    # regardless of target horizon — they tell the model "what just happened."
    # Macro features (if enabled) are appended as shared columns at the right.
    features_df, targets_1d_df, feature_names = prepare_multifeature_array(
        data, similar_stocks, start_date, end_date, macro_df=macro_df
    )

    # Forward h-day return targets (Phase 4). h=1 is identical to the Phase 2 target.
    horizon = int(get_config_value(config, "target", "horizon_days", default=1))
    if horizon < 1:
        raise ValueError(f"target.horizon_days must be >= 1, got {horizon}")
    forward_returns = compute_forward_returns(prices_df, horizon=horizon)
    past_returns = compute_past_returns(prices_df, horizon=horizon)

    # Align everything to the intersection of indices (the forward + past windows
    # drop rows at both ends; features drop rows at the start for indicator warmup).
    common_idx = features_df.index.intersection(forward_returns.index).intersection(past_returns.index)
    if common_idx.empty:
        raise ValueError("No overlapping dates between features, forward returns, and past returns")
    features_df = features_df.reindex(common_idx)
    # Absolute h-day forward returns per stock (needed for MajorityClass baseline
    # and as the raw signal before optional peer-relative demeaning).
    forward_returns_aligned = forward_returns.reindex(common_idx)[similar_stocks]
    past_returns_df = past_returns.reindex(common_idx)[similar_stocks]

    # Peer-relative target mode: subtract cross-sectional mean return for each day.
    # The model learns to predict WHICH stocks beat the peer average, removing the
    # overall market/regime direction as a confounding variable.
    target_relative = bool(get_config_value(config, "target", "relative", default=False))
    if target_relative:
        cross_sect_mean = forward_returns_aligned.mean(axis=1)
        targets_df = forward_returns_aligned.sub(cross_sect_mean, axis=0)
        logger.info(
            f"Peer-relative targets: target = stock_return − peer_mean "
            f"(horizon={horizon}d). Regime-neutral prediction task."
        )
    else:
        targets_df = forward_returns_aligned
    prices_df = prices_df.reindex(common_idx)

    n_features = len(FEATURES_PER_STOCK)
    # n_macro: number of shared macro columns appended at the right of features_df.
    # Derived directly from features_df shape so it's always consistent.
    n_macro = features_df.shape[1] - len(similar_stocks) * n_features
    if n_macro < 0:
        n_macro = 0  # safety guard
    if n_macro > 0:
        logger.info(
            f"Macro features detected: {n_macro} shared columns "
            f"(VIX / SP500). StockLSTM input_size = "
            f"{len(similar_stocks)} × {n_features} + {n_macro} = "
            f"{len(similar_stocks) * n_features + n_macro}"
        )
    logger.info(
        f"Multi-feature inputs: features {features_df.shape}, targets {targets_df.shape}, "
        f"{n_features} features per stock, {n_macro} macro features, "
        f"target horizon = {horizon} day(s)"
    )

    # Time-ordered split (train_end_date separates train+val from OOS).
    train_end_date = pd.Timestamp(get_config_value(config, "data", "train_end_date"))
    train_full_mask = features_df.index < train_end_date
    n_train_full = int(train_full_mask.sum())
    n_oos = len(features_df) - n_train_full

    if n_oos < get_config_value(config, "walk_forward", "min_oos_days", default=60):
        raise ValueError(
            f"OOS slice has only {n_oos} days but min_oos_days="
            f"{get_config_value(config, 'walk_forward', 'min_oos_days')}; widen "
            "end_date or shrink train_end_date."
        )

    val_days = get_config_value(config, "training", "validation_days", default=120)
    n_train, n_val = split_train_val(n_train_full, val_days)
    logger.info(
        f"Split: train={n_train} rows, val={n_val} rows, oos={n_oos} rows"
    )

    # Target type: 'log_returns' (regression) or 'direction' (binary classification).
    target_type = get_config_value(config, "target", "type", default="log_returns")
    direction_mode = (target_type == "direction")
    if direction_mode:
        logger.info("Target mode: DIRECTION (binary up/down classification, BCEWithLogits)")
    else:
        logger.info(f"Target mode: LOG_RETURNS (regression, horizon={horizon} days)")

    # Two scalers: inputs (features) and outputs (returns) live in different spaces.
    features_arr = features_df.values
    targets_arr = targets_df.values
    scaled_features, input_scaler = fit_scaler_on_train(features_arr, n_train, scaler_type="standard")

    if direction_mode:
        # Targets become binary labels (1=up, 0=down). No output scaler needed —
        # the model outputs logits which are converted to ±signs at eval time.
        scaled_targets = (targets_arr > 0).astype(np.float32)
        output_scaler = None
    else:
        scaled_targets, output_scaler = fit_scaler_on_train(targets_arr, n_train, scaler_type="standard")
    # Keep the legacy name 'scaler' = output_scaler so model save metadata is meaningful.
    scaler = output_scaler

    # ---------- Build sequences ----------
    seq_length = get_config_value(config, "model", "sequence_length")
    X_all, y_all = create_sequences_xy(scaled_features, scaled_targets, seq_length)
    target_dates = features_df.index[seq_length:]

    train_mask = target_dates < features_df.index[n_train]
    val_mask = (target_dates >= features_df.index[n_train]) & (
        target_dates < features_df.index[n_train_full]
    )
    oos_mask = target_dates >= features_df.index[n_train_full]

    X_train = torch.tensor(X_all[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y_all[train_mask], dtype=torch.float32)
    X_val = torch.tensor(X_all[val_mask], dtype=torch.float32)
    y_val = torch.tensor(y_all[val_mask], dtype=torch.float32)
    X_oos_scaled = X_all[oos_mask]
    y_oos_scaled = y_all[oos_mask]
    oos_dates = target_dates[oos_mask]
    logger.info(
        f"Sequences: train={len(X_train)}, val={len(X_val)}, oos={len(X_oos_scaled)} | "
        f"X input dim={X_train.shape[-1]}, y output dim={y_train.shape[-1]}"
    )

    # OOS arrays in UNSCALED space — features for LSTMs, h-day past returns for baselines.
    X_oos_features_unscaled = (X_oos_scaled * input_scaler.scale_) + input_scaler.mean_
    if direction_mode:
        # In direction mode y_oos are binary labels (0/1). For metrics we need
        # actual forward returns so the directional-accuracy calculation makes
        # sense. targets_arr[seq_length:] aligns with target_dates (same start).
        y_oos_unscaled = targets_arr[seq_length:][oos_mask]
    else:
        y_oos_unscaled = output_scaler.inverse_transform(y_oos_scaled)

    # Baselines see a sequence of past h-day returns (one per time-step).
    # `past_returns_df[t]` is `log(p[t-1] / p[t-1-h])` — the most recent h-day
    # return observable at decision day t-1, naturally lagging the X window.
    past_X_all, _ = create_sequences_xy(
        past_returns_df.values, past_returns_df.values, seq_length
    )
    X_oos_returns_unscaled = past_X_all[oos_mask]

    # Anchor prices: actual price on the day BEFORE each target date.
    # Use raw prices_df.shift(1) (1-day shift) so anchor + exp(forward h-day return)
    # reconstructs the price h days into the future from the anchor.
    anchor_prices = prices_df.shift(1).loc[oos_dates, similar_stocks].values

    # Transfer-learning pretrain/finetune split lives INSIDE the train slice.
    tl_split = get_config_value(config, "training", "transfer_learning_split", default=20)
    if tl_split >= len(X_train):
        tl_split = max(1, len(X_train) // 10)
        logger.warning(
            f"transfer_learning_split too large; using {tl_split} instead"
        )
    X_pretrain = X_train[:-tl_split]
    y_pretrain = y_train[:-tl_split]
    X_finetune = X_train[-tl_split:]
    y_finetune = y_train[-tl_split:]

    num_stocks = len(similar_stocks)
    # StockLSTM (Model 0/0f): all per-stock features + shared macro columns.
    # SingleFeatureLSTM (Model 1/1f): per-stock features + macro appended per forward pass.
    # Both use input_size that includes macro so the models can use regime context.
    multi_input_size = num_stocks * n_features + n_macro
    model_config = dict(get_config_value(config, "model"))  # mutable copy
    # Part 6 Phase 1B: apply small-model overrides when profile is selected.
    profile = model_config.get("profile", "default")
    if profile == "small":
        overrides = model_config.get("small_overrides", {}) or {}
        for k, v in overrides.items():
            model_config[k] = v
        logger.info(f"Applied small-model profile overrides: {overrides}")
    direction_loss_weight = float(model_config.get("direction_loss_weight", 0.0))
    if direction_loss_weight > 0:
        logger.info(f"Multi-task loss enabled: direction_loss_weight={direction_loss_weight}")
    if direction_mode:
        direction_loss_weight = 0.0  # direction_mode replaces the aux loss entirely
        logger.info("direction_mode=True: trainers will use BCEWithLogitsLoss as primary objective")
    es_enabled = get_config_value(config, "training", "early_stopping", "enabled", default=True)
    es_patience = get_config_value(config, "training", "early_stopping", "patience", default=10)
    es_monitor  = get_config_value(config, "training", "early_stopping", "monitor",  default="val_mse")
    logger.info(f"Early stopping: enabled={es_enabled}, patience={es_patience}, monitor={es_monitor}")

    # ---------- Training ----------
    losses_per_model: Dict[str, Dict[str, List[float]]] = {}
    best_epoch_per_model: Dict[str, int] = {}

    # Model 0
    logger.info("=" * 70)
    logger.info("Training Model 0: Multi-feature LSTM (Standard)")
    logger.info("=" * 70)
    model_0 = StockLSTM(
        input_size=multi_input_size,
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        output_size=num_stocks,
        dropout=model_config["dropout"],
    ).to(device)
    trainer_0 = LSTMTrainer(
        model_0, device,
        learning_rate=model_config["learning_rate"],
        scheduler_step_size=model_config["scheduler"]["step_size"],
        scheduler_gamma=model_config["scheduler"]["gamma"],
        direction_loss_weight=direction_loss_weight,
    )
    trainer_0.direction_mode = direction_mode
    result_0 = trainer_0.train(
        X_train, y_train, num_epochs=model_config["num_epochs"], log_interval=10,
        X_val=X_val, y_val=y_val,
        early_stopping_patience=es_patience, early_stopping_enabled=es_enabled,
        monitor=es_monitor,
    )
    losses_per_model["Model 0"] = result_0
    best_epoch_per_model["Model 0"] = result_0["best_epoch"] or len(result_0["train_losses"])

    # Model 0f
    logger.info("=" * 70)
    logger.info("Training Model 0f: Multi-feature LSTM (Transfer Learning)")
    logger.info("=" * 70)
    model_0f = StockLSTM(
        input_size=multi_input_size,
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        output_size=num_stocks,
        dropout=model_config["dropout"],
    ).to(device)
    trainer_0f = LSTMTrainer(
        model_0f, device,
        learning_rate=model_config["learning_rate"],
        direction_loss_weight=direction_loss_weight,
    )
    trainer_0f.direction_mode = direction_mode
    pre_0f = trainer_0f.train(
        X_pretrain, y_pretrain, num_epochs=model_config["num_epochs"], log_interval=10,
        X_val=X_val, y_val=y_val,
        early_stopping_patience=es_patience, early_stopping_enabled=es_enabled,
        monitor=es_monitor,
    )
    ft_0f = trainer_0f.freeze_layers_and_finetune(
        num_layers_to_freeze=model_config["transfer_learning"]["freeze_layers"],
        X_finetune=X_finetune, y_finetune=y_finetune,
        num_epochs=model_config["num_epochs"], log_interval=10,
        X_val=X_val, y_val=y_val,
        early_stopping_patience=es_patience, early_stopping_enabled=es_enabled,
        monitor=es_monitor,
    )
    losses_per_model["Model 0f"] = {
        "train_losses": pre_0f["train_losses"] + ft_0f["train_losses"],
        "val_losses": pre_0f["val_losses"] + ft_0f["val_losses"],
    }
    best_epoch_per_model["Model 0f"] = ft_0f["best_epoch"] or pre_0f["best_epoch"] or 0

    # Model 1
    logger.info("=" * 70)
    logger.info("Training Model 1: Single-feature LSTM (Per-stock)")
    logger.info("=" * 70)
    model_1 = SingleFeatureLSTM(
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        input_size=n_features + n_macro,
    ).to(device)
    trainer_1 = SingleFeatureTrainer(
        model_1, device,
        learning_rate=model_config["learning_rate_single_feature"],
        direction_loss_weight=direction_loss_weight,
    )
    trainer_1.direction_mode = direction_mode
    result_1 = trainer_1.train_per_stock(
        X_train, y_train, num_epochs=model_config["num_epochs_single_feature"],
        num_stocks=num_stocks, log_interval=2,
        X_val=X_val, y_val=y_val,
        early_stopping_patience=es_patience, early_stopping_enabled=es_enabled,
        n_features_per_stock=n_features, n_macro=n_macro,
        monitor=es_monitor,
    )
    losses_per_model["Model 1"] = result_1
    best_epoch_per_model["Model 1"] = result_1["best_epoch"] or len(result_1["train_losses"])

    # Model 1f
    logger.info("=" * 70)
    logger.info("Training Model 1f: Single-feature LSTM (Transfer Learning)")
    logger.info("=" * 70)
    model_1f = SingleFeatureLSTM(
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        input_size=n_features + n_macro,
    ).to(device)
    trainer_1f = SingleFeatureTrainer(
        model_1f, device,
        learning_rate=model_config["learning_rate_single_feature"],
        direction_loss_weight=direction_loss_weight,
    )
    trainer_1f.direction_mode = direction_mode
    pre_1f = trainer_1f.train_per_stock(
        X_pretrain, y_pretrain, num_epochs=model_config["num_epochs_single_feature"],
        num_stocks=num_stocks, log_interval=2,
        X_val=X_val, y_val=y_val,
        early_stopping_patience=es_patience, early_stopping_enabled=es_enabled,
        n_features_per_stock=n_features, n_macro=n_macro,
        monitor=es_monitor,
    )
    model_1f.freeze_layers(model_config["transfer_learning"]["freeze_layers"])
    ft_1f = trainer_1f.train_per_stock(
        X_finetune, y_finetune, num_epochs=model_config["num_epochs_single_feature"],
        num_stocks=num_stocks, log_interval=2,
        X_val=X_val, y_val=y_val,
        early_stopping_patience=es_patience, early_stopping_enabled=es_enabled,
        n_features_per_stock=n_features, n_macro=n_macro,
        monitor=es_monitor,
    )
    losses_per_model["Model 1f"] = {
        "train_losses": pre_1f["train_losses"] + ft_1f["train_losses"],
        "val_losses": pre_1f["val_losses"] + ft_1f["val_losses"],
    }
    best_epoch_per_model["Model 1f"] = ft_1f["best_epoch"] or pre_1f["best_epoch"] or 0

    # ---------- Save trained models ----------
    logger.info("Saving trained models...")
    models_dir = get_config_value(config, "persistence", "models_dir", default="./saved_models")
    save_model(model_0, f"{models_dir}/model_0.pt", scaler=scaler)
    save_model(model_0f, f"{models_dir}/model_0f.pt", scaler=scaler)
    save_model(model_1, f"{models_dir}/model_1.pt", scaler=scaler)
    save_model(model_1f, f"{models_dir}/model_1f.pt", scaler=scaler)

    # ---------- Walk-forward ----------
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "per_stock").mkdir(parents=True, exist_ok=True)

    wf_cfg = get_config_value(config, "walk_forward")
    window_days = wf_cfg["window_days"]
    stride_days = wf_cfg["stride_days"]

    lstm_predictors = [
        LSTMPredictor(model_0, input_scaler, device, "Model 0", output_scaler=output_scaler),
        LSTMPredictor(model_0f, input_scaler, device, "Model 0f", output_scaler=output_scaler),
        SingleFeaturePredictor(
            model_1, input_scaler, device, "Model 1",
            output_scaler=output_scaler, n_features_per_stock=n_features,
            n_macro_features=n_macro,
        ),
        SingleFeaturePredictor(
            model_1f, input_scaler, device, "Model 1f",
            output_scaler=output_scaler, n_features_per_stock=n_features,
            n_macro_features=n_macro,
        ),
    ]
    # In direction mode each predictor outputs ±1e-4 signs rather than returns.
    if direction_mode:
        for p in lstm_predictors:
            p.direction_mode = True

    # ---------- Val-loss weighting for ensemble ----------
    # Use each model's best validation loss (BCE in direction_mode, MSE in
    # log_returns mode) recorded during training.  Lower loss = better model
    # = more ensemble weight.  We store as negative loss so that
    # EnsembleWeighted's softmax(score) gives more weight to the best model.
    #
    # Why not val RMSE on prices?  In direction_mode all predictors output
    # ±1e-4 signs, so their price-RMSE is nearly identical — the ensemble
    # degrades to equal weights.  The trainer's best_val is computed on raw
    # logits (BCE) or scaled returns (MSE), which cleanly differentiates
    # models in either mode.
    _best_val_by_name = {
        "Model 0":  result_0.get("best_val"),
        "Model 0f": ft_0f.get("best_val") or pre_0f.get("best_val"),
        "Model 1":  result_1.get("best_val"),
        "Model 1f": ft_1f.get("best_val") or pre_1f.get("best_val"),
    }
    # When monitor='val_da': best_val is already a DA% (higher = better) → use directly.
    # When monitor='val_mse': best_val is a loss (lower = better) → negate for weighting.
    logger.info(f"Ensemble weighting using monitor='{es_monitor}':")
    for p in lstm_predictors:
        bv = _best_val_by_name.get(p.name)
        if bv is not None:
            p.val_da = bv if es_monitor == "val_da" else -bv
            logger.info(f"  {p.name}: best val metric = {bv:.4f}  (weight score = {p.val_da:.4f})")
        else:
            p.val_da = None
            logger.warning(f"  {p.name}: no val metric recorded → equal weight fallback")

    # ---------- XGBoost per-stock direction classifier ----------
    xgb_predictors = []
    if HAS_XGB:
        logger.info("=" * 70)
        logger.info("Training XGBoost per-stock direction classifiers…")
        logger.info("=" * 70)

        # Training data in unscaled feature space (same arrays the LSTM uses).
        X_train_unscaled = (X_all[train_mask] * input_scaler.scale_) + input_scaler.mean_
        y_train_returns   = targets_arr[seq_length:][train_mask]

        # Strategy A — last time-step only (fast, interpretable)
        xgb_last = XGBoostPredictor(
            input_scaler, num_stocks, n_features, n_macro,
            name="XGBoost", use_last_step_only=True,
        )
        xgb_last.fit(X_train_unscaled, y_train_returns)
        xgb_predictors.append(xgb_last)
        logger.info("XGBoost (last-step) trained.")

        # Strategy B — full sequence flattened (captures temporal evolution)
        xgb_seq = XGBoostPredictor(
            input_scaler, num_stocks, n_features, n_macro,
            name="XGBoost-Seq", use_last_step_only=False,
        )
        xgb_seq.fit(X_train_unscaled, y_train_returns)
        xgb_predictors.append(xgb_seq)
        logger.info("XGBoost (sequence) trained.")
    else:
        logger.warning("xgboost not installed — skipping XGBoost predictors. Run: pip install xgboost")

    predictors = lstm_predictors + xgb_predictors + [
        EnsembleWeighted(lstm_predictors, name="Ensemble"),
        NaivePersistence(),
        # MajorityClass uses absolute forward returns even in relative mode:
        # peer-relative means are ≈0 by construction (sum to 0 across peers),
        # which would create a 0-prediction degenerate baseline.
        MajorityClass(forward_returns_aligned.iloc[:n_train].mean().values),
    ]

    wf_frames = []
    pooled_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    per_stock_frames: Dict[str, pd.DataFrame] = {}

    for predictor in predictors:
        name = predictor.name if hasattr(predictor, "name") else type(predictor).__name__
        logger.info(f"Walk-forward evaluating: {name}")
        windows = walk_forward_predict(
            predictor,
            X_oos_features_unscaled,
            y_oos_unscaled,
            anchor_prices,
            oos_dates,
            similar_stocks,
            window_days=window_days,
            stride_days=stride_days,
            X_oos_returns=X_oos_returns_unscaled,
        )
        wf_frames.append(windows_to_frame(windows, name))
        pooled_by_model[name] = pool_windows(windows)
        per_stock_frames[name] = pooled_per_stock(pooled_by_model[name], similar_stocks)
        safe = name.replace(" ", "_").lower()
        per_stock_frames[name].to_csv(OUTPUTS_DIR / "per_stock" / f"{safe}.csv")

    wf_results = pd.concat(wf_frames, ignore_index=True)
    wf_results.to_csv(OUTPUTS_DIR / "walk_forward_results.csv", index=False)

    # ---------- Summary metrics: mean / std / min / max per model ----------
    agg = (
        wf_results.groupby("model")[["RMSE", "MAPE", "DirectionalAccuracy"]]
        .agg(["mean", "std", "min", "max"])
    )
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg.to_csv(OUTPUTS_DIR / "metrics.csv")

    print()
    da_mean_col = "DirectionalAccuracy_mean"
    da_std_col = "DirectionalAccuracy_std"
    print(
        agg[["RMSE_mean", "MAPE_mean", da_mean_col, da_std_col]]
        .rename(
            columns={
                "RMSE_mean": "RMSE",
                "MAPE_mean": "MAPE%",
                da_mean_col: "DirAcc%",
                da_std_col: "±std",
            }
        )
        .to_string(float_format=lambda v: f"{v:8.3f}")
    )
    print()

    best_da = agg[da_mean_col].idxmax()
    naive_da = agg.loc["Naive", da_mean_col] if "Naive" in agg.index else float("nan")
    if pd.notna(naive_da):
        lift = agg.loc[best_da, da_mean_col] - naive_da
        logger.info(
            f"Best success rate: {best_da} ({agg.loc[best_da, da_mean_col]:.1f}%) "
            f"vs Naive ({naive_da:.1f}%) → lift {lift:+.1f}pp"
        )

    # ---------- Long-format predictions for the dashboard ----------
    pred_rows = []
    for name, pooled in pooled_by_model.items():
        for row_idx in range(len(pooled["dates"])):
            date = pd.Timestamp(pooled["dates"][row_idx])
            for col_idx, ticker in enumerate(similar_stocks):
                pred_rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "model": name,
                        "actual": float(pooled["true_prices"][row_idx, col_idx]),
                        "predicted": float(pooled["pred_prices"][row_idx, col_idx]),
                    }
                )
    pd.DataFrame(pred_rows).to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

    # ---------- Losses (train + val per epoch, long format) ----------
    loss_rows = []
    for name, info in losses_per_model.items():
        for epoch, loss in enumerate(info.get("train_losses", []), start=1):
            loss_rows.append({"model": name, "epoch": epoch, "split": "train", "loss": float(loss)})
        for epoch, loss in enumerate(info.get("val_losses", []), start=1):
            loss_rows.append({"model": name, "epoch": epoch, "split": "val", "loss": float(loss)})
    pd.DataFrame(loss_rows).to_csv(OUTPUTS_DIR / "losses.csv", index=False)

    # ---------- Plots ----------
    # Aggregate dirAcc per model for the comparison bar (with error bars).
    per_model_da = {m: agg.loc[m, da_mean_col] for m in agg.index}
    per_model_da_std = {m: agg.loc[m, da_std_col] for m in agg.index}
    plot_directional_accuracy(per_model_da, std_per_model=per_model_da_std)

    # Walk-forward over time.
    da_by_window = {
        name: wf_results[wf_results["model"] == name][["window_start", "DirectionalAccuracy"]]
        for name in agg.index
    }
    plot_walk_forward_accuracy(da_by_window)

    # Per-stock metrics for each LSTM model.
    for name in ("Model 0", "Model 0f", "Model 1", "Model 1f"):
        if name in per_stock_frames:
            plot_per_stock_metrics(per_stock_frames[name], name)

    # Loss curves (train + val per model).
    plot_loss_curves(losses_per_model)

    # Plain RMSE/MSE comparison kept for backward-compat with the existing plot.
    plot_model_comparison(
        {
            m: {"MSE": agg.loc[m, "RMSE_mean"] ** 2, "RMSE": agg.loc[m, "RMSE_mean"]}
            for m in agg.index
        }
    )

    # Per-stock prediction lines on the pooled OOS slice for a couple of tickers.
    viz_config = get_config_value(config, "visualization")
    plot_stocks = viz_config.get("plot_stocks", similar_stocks[:2])

    pooled_index = pd.DatetimeIndex(pooled_by_model[next(iter(pooled_by_model))]["dates"])
    pooled_actuals_df = pd.DataFrame(
        pooled_by_model[next(iter(pooled_by_model))]["true_prices"],
        index=pooled_index,
        columns=similar_stocks,
    )
    predictions_dict = {
        name: pooled_by_model[name]["pred_prices"] for name in ("Model 0", "Model 0f", "Model 1", "Model 1f")
    }
    n_test_days = min(60, len(pooled_index))
    plot_multiple_stocks(
        pooled_actuals_df,
        plot_stocks,
        predictions_dict,
        num_test_days=n_test_days,
        figsize=(viz_config["figure_size"]["width"], viz_config["figure_size"]["height"]),
        y_limit_bottom=viz_config.get("y_limit_bottom"),
    )

    logger.info("=" * 70)
    logger.info("EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
