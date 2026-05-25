# Stock Price Prediction System

A modular, production-ready stock price prediction system using LSTM neural networks and technical indicators.

## Overview

This system predicts stock prices using:
- **LSTM Neural Networks**: Deep learning for time series prediction
- **Technical Indicators**: RSI, Momentum, Moving Averages, Bollinger Bands
- **Stock Similarity Analysis**: Find similar stocks based on technical patterns
- **Transfer Learning**: Pre-train and fine-tune for better performance

## Recent Improvements

This project has been completely refactored from a monolithic 452-line script into a modular, maintainable architecture:

### ✅ Completed Improvements

1. **Modular Architecture**
   - Separated into logical modules: `data/`, `models/`, `training/`, `utils/`, `config/`
   - Eliminated 100+ lines of duplicated training code
   - Clean separation of concerns

2. **Configuration Management**
   - All hardcoded parameters moved to `config/config.yaml`
   - Easy to experiment with different settings
   - No code changes needed for parameter tuning

3. **LSTM Architecture Fixes**
   - Fixed dropout issue (was non-functional with single-layer LSTMs)
   - Properly implemented stacked LSTM architecture
   - Improved model performance and stability

4. **Error Handling & Logging**
   - Comprehensive logging throughout all modules
   - Proper error handling with informative messages
   - Configurable log levels and outputs

5. **Model Persistence**
   - Save and load trained models
   - Checkpoint support during training
   - Metadata tracking (loss, epochs, timestamps)

6. **Dependency Management**
   - `requirements.txt` with all dependencies
   - Proper `.gitignore` for version control
   - Virtual environment support

7. **Documentation & Type Hints**
   - Comprehensive docstrings for all functions
   - Python type hints for better IDE support
   - Clear examples in documentation

8. **Test Suite**
   - Unit tests for feature engineering
   - Unit tests for data preprocessing
   - Unit tests for LSTM models
   - Run with: `pytest tests/`

## Project Structure

```
stockrecommendation/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── __init__.py
│   ├── download.py              # Stock data download
│   ├── features.py              # Technical indicators
│   ├── similarity.py            # Stock similarity analysis
│   └── preprocessing.py         # Data preprocessing
├── models/
│   ├── __init__.py
│   ├── lstm.py                  # LSTM model architecture
│   └── persistence.py           # Model save/load
├── training/
│   ├── __init__.py
│   └── trainer.py               # Unified training module
├── utils/
│   ├── __init__.py
│   ├── config_loader.py         # Configuration loading
│   ├── device.py                # Device selection (MPS/CUDA/CPU)
│   ├── logger.py                # Logging setup
│   └── visualization.py         # Plotting utilities
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   └── test_models.py
├── main.py                      # Main entry point (NEW)
├── main_old.py                  # Original monolithic code (BACKUP)
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Simply run the main script:

```bash
python main.py
```

This will:
1. Download stock data from Yahoo Finance
2. Calculate technical indicators
3. Find similar stocks to a base ticker
4. Train 4 different LSTM models
5. Evaluate and compare performance
6. Save trained models
7. Display visualization plots

### Streamlit Dashboard

After `python main.py` finishes (it writes `outputs/metrics.csv`,
`outputs/predictions.csv`, `outputs/losses.csv`, and per-stock CSVs), launch the
interactive analyst view:

```bash
streamlit run dashboard.py
```

The dashboard shows headline KPIs (best **success rate** = directional accuracy,
best RMSE), per-model comparison charts, a per-stock drill-down with actual vs
predicted overlays for every model, and training loss curves. If `outputs/` is
empty the dashboard prints a friendly hint instead of crashing.

### Configuration

Edit `config/config.yaml` to customize:

- **Stock tickers**: Change the list of stocks to analyze
- **Date ranges**: Modify training/testing periods
- **Model hyperparameters**: Adjust learning rate, hidden size, layers, etc.
- **Technical indicators**: Configure RSI, momentum, Bollinger Bands windows
- **Training settings**: Epochs, batch size, learning rate schedule

Example:
```yaml
data:
  tickers:
    - AAPL
    - MSFT
    - GOOGL
  start_date: '2020-01-01'
  end_date: '2025-12-31'

model:
  hidden_size: 100
  num_layers: 4
  dropout: 0.2
  learning_rate: 0.001
```

### Advanced Usage

#### Load a Trained Model

```python
from models.lstm import StockLSTM
from models.persistence import load_model
from utils.device import get_device

device = get_device()
model = StockLSTM(input_size=10, hidden_size=100, num_layers=4, output_size=10)
checkpoint = load_model(model, 'saved_models/model_0.pt', device)

scaler = checkpoint['scaler']
# Use model for predictions...
```

#### Train a Custom Model

```python
from models.lstm import StockLSTM
from training.trainer import LSTMTrainer

model = StockLSTM(input_size=10, hidden_size=100, num_layers=4, output_size=10)
trainer = LSTMTrainer(model, device, learning_rate=0.001)
losses = trainer.train(X_train, y_train, num_epochs=100)
```

#### Calculate Technical Indicators

```python
from data.features import calculate_all_features

df_with_features = calculate_all_features(
    df,
    rsi_window=14,
    momentum_window=10,
    ma_window=20
)
```

## Models

The system trains and compares 4 different models:

1. **Model 0**: Multi-feature LSTM (standard training)
   - Trains on all stocks simultaneously
   - Uses all time series data

2. **Model 0f**: Multi-feature LSTM (transfer learning)
   - Pre-trains on 80% of data
   - Fine-tunes on recent 15% of data
   - Better for capturing recent trends

3. **Model 1**: Single-feature LSTM (per-stock)
   - Trains separately on each stock
   - More specialized predictions

4. **Model 1f**: Single-feature LSTM (transfer learning)
   - Combines per-stock training with transfer learning
   - Best of both approaches

## Testing

Run the test suite:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py
```

## Performance Metrics

The system trains on **log-returns** (`r_t = log(p_t / p_{t-1})`), splits time-orderedly
into train / val / out-of-sample, fits a `StandardScaler` on the train slice only,
and uses **early stopping** on validation MSE to pick the best checkpoint. Evaluation
runs **walk-forward** across many 21-day windows on the OOS slice — each metric is a
**mean ± std across windows**, not a single tail slice. Two reference predictors run
through the same pipeline:
- **Naive persistence** — predict tomorrow's return = today's return.
- **Majority class** — predict the training-set mean return every day (captures bull-market drift).

Any LSTM that doesn't beat these baselines isn't learning anything useful, just memorizing.

The system evaluates models using:
- **MSE / RMSE**: magnitude of error (lower is better), in dollars after reconstruction.
- **MAPE (Mean Absolute Percentage Error)**: error as a % of price.
- **Directional Accuracy (Success Rate)**: % of test samples where the model
  predicted the right next-day direction (up vs. down). 50% is random; >50% is
  the only number that actually matters for a recommendation system.

## Logging

Logs are written to:
- **Console**: Real-time progress updates
- **File**: `stock_prediction.log` (configurable in config.yaml)

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## GPU Support

The system automatically detects and uses:
1. **Apple Silicon (MPS)**: For M1/M2/M3 Macs
2. **NVIDIA CUDA**: For NVIDIA GPUs
3. **CPU**: Fallback if no GPU available

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you're in the project root directory and the virtual environment is activated.

### PyTorch Installation

For specific PyTorch installations (CUDA version, etc.), visit:
https://pytorch.org/get-started/locally/

### Data Download Issues

If Yahoo Finance data download fails:
- Check your internet connection
- Some tickers may be delisted or invalid
- The system will skip failed tickers and continue

## Future Enhancements

Potential improvements:
- [ ] Web interface (Flask/FastAPI)
- [ ] Database integration for caching
- [ ] Real-time prediction API
- [ ] Backtesting framework
- [ ] Ensemble methods
- [ ] Hyperparameter optimization
- [ ] Attention mechanisms

## Original Code

The original monolithic code is preserved in `main_old.py` for reference.

## License

[Add your license here]

## Contributing

[Add contribution guidelines if applicable]

## Acknowledgments

- Yahoo Finance for stock data (via yfinance)
- PyTorch for deep learning framework
- scikit-learn for preprocessing utilities
