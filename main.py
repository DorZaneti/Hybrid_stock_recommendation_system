"""
Stock Price Prediction using LSTM Neural Networks

This is the main entry point for the stock recommendation system.
It orchestrates data download, feature engineering, model training, and visualization.
"""
from typing import Dict, List
import numpy as np

# Configuration and utilities
from utils.config_loader import load_config, get_config_value
from utils.logger import setup_logger
from utils.device import get_device, set_random_seed
from utils.visualization import plot_multiple_stocks, plot_model_comparison

# Data processing
from data.download import download_stock_data
from data.features import calculate_all_features
from data.similarity import find_similar_stocks
from data.preprocessing import (
    prepare_stock_dataframe,
    scale_data,
    create_sequences,
    split_train_test,
    split_for_transfer_learning
)

# Models and training
from models.lstm import StockLSTM, SingleFeatureLSTM
from models.persistence import save_model
from training.trainer import LSTMTrainer, SingleFeatureTrainer


def main():
    """Main execution function."""

    # ========== 1. SETUP AND CONFIGURATION ==========
    print("=" * 70)
    print("STOCK PRICE PREDICTION SYSTEM")
    print("=" * 70)

    # Load configuration
    config = load_config()
    logger = setup_logger(
        __name__,
        level=get_config_value(config, 'logging', 'level', default='INFO'),
        log_file=get_config_value(config, 'logging', 'file'),
        console_output=get_config_value(config, 'logging', 'console_output', default=True)
    )

    logger.info("Starting stock price prediction system")

    # Setup device
    device = get_device(get_config_value(config, 'training', 'device_priority'))
    set_random_seed(get_config_value(config, 'training', 'random_seed', default=42), device)

    # ========== 2. DATA DOWNLOAD ==========
    logger.info("Phase 1: Downloading stock data")
    tickers = get_config_value(config, 'data', 'tickers')
    start_date = get_config_value(config, 'data', 'start_date')
    end_date = get_config_value(config, 'data', 'end_date')

    data = download_stock_data(tickers, start_date, end_date)
    logger.info(f"Downloaded data for {len(data)} stocks")

    # ========== 3. FEATURE ENGINEERING ==========
    logger.info("Phase 2: Calculating technical indicators")
    feature_config = get_config_value(config, 'features')

    for ticker in data.keys():
        data[ticker] = calculate_all_features(
            data[ticker],
            rsi_window=feature_config['rsi']['window'],
            momentum_window=feature_config['momentum']['window'],
            ma_window=feature_config['moving_average']['window'],
            bb_window=feature_config['bollinger_bands']['window'],
            bb_std=feature_config['bollinger_bands']['std_multiplier']
        )

    # ========== 4. STOCK SIMILARITY ANALYSIS ==========
    logger.info("Phase 3: Finding similar stocks")
    sim_config = get_config_value(config, 'similarity')
    base_ticker = sim_config['base_ticker']
    num_similar = sim_config['num_similar_stocks']

    similar_stocks = find_similar_stocks(data, base_ticker, num_similar)
    logger.info(f"Similar stocks to {base_ticker}: {similar_stocks}")

    # ========== 5. DATA PREPROCESSING ==========
    logger.info("Phase 4: Preparing data for training")
    train_start = get_config_value(config, 'data', 'train_start_date')
    train_end = get_config_value(config, 'data', 'train_end_date')

    df = prepare_stock_dataframe(data, similar_stocks, train_start, train_end)
    logger.info(f"Prepared DataFrame shape: {df.shape}")

    # Scale data
    scaled_data, scaler = scale_data(df)

    # Create sequences
    seq_length = get_config_value(config, 'model', 'sequence_length')
    X, y = create_sequences(scaled_data, seq_length)

    # Split data
    test_days = get_config_value(config, 'training', 'test_days')
    tl_split = get_config_value(config, 'training', 'transfer_learning_split')

    X_train, y_train, X_test, y_test = split_train_test(X, y, test_days)
    X_pretrain, y_pretrain, X_finetune, y_finetune, _, _ = split_for_transfer_learning(
        X, y, test_days, tl_split
    )

    num_stocks = len(similar_stocks)
    logger.info(f"Training on {num_stocks} stocks")

    # ========== 6. MODEL TRAINING ==========
    model_config = get_config_value(config, 'model')
    results = {}

    # Model 0: Multi-feature standard training
    logger.info("=" * 70)
    logger.info("Training Model 0: Multi-feature LSTM (Standard)")
    logger.info("=" * 70)

    model_0 = StockLSTM(
        input_size=num_stocks,
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        output_size=num_stocks,
        dropout=model_config['dropout']
    ).to(device)

    trainer_0 = LSTMTrainer(
        model_0,
        device,
        learning_rate=model_config['learning_rate'],
        scheduler_step_size=model_config['scheduler']['step_size'],
        scheduler_gamma=model_config['scheduler']['gamma']
    )

    trainer_0.train(X_train, y_train, num_epochs=model_config['num_epochs'], log_interval=10)
    mse_0, rmse_0, outputs_0 = trainer_0.evaluate(X_test, y_test)

    predicted_prices_0 = scaler.inverse_transform(outputs_0.cpu().detach().numpy())
    results['Model 0'] = {'MSE': mse_0, 'RMSE': rmse_0, 'predictions': predicted_prices_0}

    # Model 0f: Multi-feature transfer learning
    logger.info("=" * 70)
    logger.info("Training Model 0f: Multi-feature LSTM (Transfer Learning)")
    logger.info("=" * 70)

    model_0f = StockLSTM(
        input_size=num_stocks,
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        output_size=num_stocks,
        dropout=model_config['dropout']
    ).to(device)

    trainer_0f = LSTMTrainer(model_0f, device, learning_rate=model_config['learning_rate'])

    # Pretrain
    logger.info("Pretraining on larger dataset...")
    trainer_0f.train(X_pretrain, y_pretrain, num_epochs=model_config['num_epochs'], log_interval=10)

    # Fine-tune
    logger.info("Fine-tuning on recent data...")
    trainer_0f.freeze_layers_and_finetune(
        num_layers_to_freeze=model_config['transfer_learning']['freeze_layers'],
        X_finetune=X_finetune,
        y_finetune=y_finetune,
        num_epochs=model_config['num_epochs'],
        log_interval=10
    )

    mse_0f, rmse_0f, outputs_0f = trainer_0f.evaluate(X_test, y_test)
    predicted_prices_0f = scaler.inverse_transform(outputs_0f.cpu().detach().numpy())
    results['Model 0f'] = {'MSE': mse_0f, 'RMSE': rmse_0f, 'predictions': predicted_prices_0f}

    # Model 1: Single-feature per-stock training
    logger.info("=" * 70)
    logger.info("Training Model 1: Single-feature LSTM (Per-stock)")
    logger.info("=" * 70)

    model_1 = SingleFeatureLSTM(
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)

    trainer_1 = SingleFeatureTrainer(
        model_1,
        device,
        learning_rate=model_config['learning_rate_single_feature']
    )

    trainer_1.train_per_stock(
        X_train,
        y_train,
        num_epochs=model_config['num_epochs_single_feature'],
        num_stocks=num_stocks,
        log_interval=2
    )

    mse_1, rmse_1, outputs_1 = trainer_1.evaluate_per_stock(X_test, y_test, num_stocks)
    predicted_prices_1 = scaler.inverse_transform(outputs_1.cpu().detach().numpy())
    results['Model 1'] = {'MSE': mse_1, 'RMSE': rmse_1, 'predictions': predicted_prices_1}

    # Model 1f: Single-feature transfer learning
    logger.info("=" * 70)
    logger.info("Training Model 1f: Single-feature LSTM (Transfer Learning)")
    logger.info("=" * 70)

    model_1f = SingleFeatureLSTM(
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)

    trainer_1f = SingleFeatureTrainer(
        model_1f,
        device,
        learning_rate=model_config['learning_rate_single_feature']
    )

    # Pretrain
    logger.info("Pretraining per-stock model...")
    trainer_1f.train_per_stock(
        X_pretrain,
        y_pretrain,
        num_epochs=model_config['num_epochs_single_feature'],
        num_stocks=num_stocks,
        log_interval=2
    )

    # Fine-tune
    logger.info("Fine-tuning per-stock model...")
    model_1f.freeze_layers(model_config['transfer_learning']['freeze_layers'])
    trainer_1f.train_per_stock(
        X_finetune,
        y_finetune,
        num_epochs=model_config['num_epochs_single_feature'],
        num_stocks=num_stocks,
        log_interval=2
    )

    mse_1f, rmse_1f, outputs_1f = trainer_1f.evaluate_per_stock(X_test, y_test, num_stocks)
    predicted_prices_1f = scaler.inverse_transform(outputs_1f.cpu().detach().numpy())
    results['Model 1f'] = {'MSE': mse_1f, 'RMSE': rmse_1f, 'predictions': predicted_prices_1f}

    # ========== 7. SAVE MODELS ==========
    logger.info("Saving trained models...")
    models_dir = get_config_value(config, 'persistence', 'models_dir', default='./saved_models')

    save_model(model_0, f"{models_dir}/model_0.pt", metadata={'MSE': mse_0, 'RMSE': rmse_0}, scaler=scaler)
    save_model(model_0f, f"{models_dir}/model_0f.pt", metadata={'MSE': mse_0f, 'RMSE': rmse_0f}, scaler=scaler)
    save_model(model_1, f"{models_dir}/model_1.pt", metadata={'MSE': mse_1, 'RMSE': rmse_1}, scaler=scaler)
    save_model(model_1f, f"{models_dir}/model_1f.pt", metadata={'MSE': mse_1f, 'RMSE': rmse_1f}, scaler=scaler)

    # ========== 8. VISUALIZATION ==========
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    for model_name, metrics in results.items():
        logger.info(f"{model_name}: MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}")

    # Plot comparison
    metrics_for_plot = {name: {'MSE': m['MSE'], 'RMSE': m['RMSE']} for name, m in results.items()}
    plot_model_comparison(metrics_for_plot)

    # Plot predictions for selected stocks
    viz_config = get_config_value(config, 'visualization')
    plot_stocks = viz_config.get('plot_stocks', similar_stocks[:2])

    predictions_dict = {
        'Model 0': predicted_prices_0,
        'Model 0f': predicted_prices_0f,
        'Model 1': predicted_prices_1,
        'Model 1f': predicted_prices_1f
    }

    plot_multiple_stocks(
        df,
        plot_stocks,
        predictions_dict,
        num_test_days=test_days,
        figsize=(viz_config['figure_size']['width'], viz_config['figure_size']['height']),
        y_limit_bottom=viz_config.get('y_limit_bottom')
    )

    logger.info("=" * 70)
    logger.info("EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
