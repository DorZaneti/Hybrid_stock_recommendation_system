# STOCK RECOMMENDATION SYSTEM - KNOWLEDGE BASE

**Version:** 1.0
**Last Updated:** 2025-12-23
**Main File:** `main.py` (452 lines)

---

## TABLE OF CONTENTS

1. [Architecture Overview](#1-architecture-overview)
2. [Setup & Installation](#2-setup--installation)
3. [API & Function Reference](#3-api--function-reference)
4. [Data Structures](#4-data-structures)
5. [Model Architecture](#5-model-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Common Troubleshooting](#7-common-troubleshooting)
8. [Configuration Reference](#8-configuration-reference)
9. [Execution Flow](#9-execution-flow)
10. [Performance Benchmarks](#10-performance-benchmarks)

---

## 1. ARCHITECTURE OVERVIEW

### 1.1 System Purpose

The Stock Recommendation System is a deep learning-based stock price forecasting tool that:
- Downloads historical stock data from Yahoo Finance
- Calculates technical indicators (RSI, Momentum, Bollinger Bands, Moving Averages)
- Identifies similar stocks using feature-based Euclidean distance
- Trains 4 different LSTM neural network variants
- Predicts future stock prices (5-day forecast)
- Compares model performance through visualization

### 1.2 Architecture Type

**Monolithic single-file application** - All functionality contained in `main.py`

### 1.3 Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Language** | Python | 3.11.2 |
| **Deep Learning** | PyTorch | 2.9.1 |
| **Data Acquisition** | yfinance | 0.2.66 |
| **Data Processing** | pandas | 2.3.3 |
| **Numerical Computing** | numpy | 2.4.0 |
| **Machine Learning** | scikit-learn | 1.8.0 |
| **Scientific Computing** | scipy | 1.16.3 |
| **Visualization** | matplotlib | 3.10.8 |
| **Web Scraping** | beautifulsoup4 | 4.14.3 |
| **HTTP Client** | requests | 2.32.5 |

### 1.4 Compute Backend

The system automatically selects the optimal compute device in this priority order:

1. **Apple Silicon (MPS)** - M1/M2/M3 Mac GPUs
2. **NVIDIA CUDA** - CUDA-enabled GPUs
3. **CPU** - Fallback for systems without GPU

**Code Location:** Lines 21-28

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

### 1.5 System Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA ACQUISITION                                   │
│ - Download 100 stocks (2020-2025) via yfinance             │
│ - Store in dictionary: {ticker: DataFrame}                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: FEATURE ENGINEERING                                │
│ - Calculate 5 technical indicators per stock                │
│ - RSI (14-day), Momentum (10-day), MA (20-day)             │
│ - Bollinger Bands (20-day, 2σ)                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: SIMILARITY ANALYSIS                                │
│ - Base ticker: AMZN                                         │
│ - Calculate pairwise Euclidean distances                    │
│ - Select top 10 most similar stocks                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: DATA PREPROCESSING                                 │
│ - Filter to 2020-01-01 to 2023-01-01                       │
│ - MinMaxScaler normalization [0,1]                          │
│ - Create 8-day sliding window sequences                     │
│ - Split: Train (all-5), Test (last 5)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: MODEL TRAINING (4 variants)                        │
│ Model 0:  Multi-feature, standard (100 epochs)              │
│ Model 0f: Multi-feature, transfer learning (100+100)        │
│ Model 1:  Single-feature, per-stock (10 epochs)             │
│ Model 1f: Single-feature, transfer learning (10+10)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 6: PREDICTION & VISUALIZATION                         │
│ - Generate 5-day forecasts for all 10 stocks               │
│ - Plot comparisons for UPS and KMB                          │
│ - Calculate MSE and RMSE metrics                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. SETUP & INSTALLATION

### 2.1 Prerequisites

- **Python:** Version 3.11.2 or compatible
- **Operating System:** macOS (optimized for Apple Silicon), Linux, Windows
- **Hardware:**
  - Minimum: 8GB RAM
  - Recommended: 16GB RAM + GPU (Apple Silicon or NVIDIA CUDA)
- **Internet Connection:** Required for downloading stock data

### 2.2 Virtual Environment Setup

The project uses a Python virtual environment located at `.venv/`

**Create virtual environment:**
```bash
python3.11 -m venv .venv
```

**Activate virtual environment:**
```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 2.3 Dependency Installation

**Install all dependencies:**
```bash
pip install torch==2.9.1
pip install yfinance==0.2.66
pip install pandas==2.3.3
pip install numpy==2.4.0
pip install scikit-learn==1.8.0
pip install scipy==1.16.3
pip install matplotlib==3.10.8
pip install beautifulsoup4==4.14.3
pip install requests==2.32.5
pip install curl-cffi==0.14.0
pip install websockets==15.0.1
```

**Alternative: Create requirements.txt**
```text
torch==2.9.1
yfinance==0.2.66
pandas==2.3.3
numpy==2.4.0
scikit-learn==1.8.0
scipy==1.16.3
matplotlib==3.10.8
beautifulsoup4==4.14.3
requests==2.32.5
curl-cffi==0.14.0
websockets==15.0.1
```

```bash
pip install -r requirements.txt
```

### 2.4 Running the System

**Basic execution:**
```bash
python main.py
```

**Expected runtime:** 5-15 minutes depending on:
- Network speed (data download)
- CPU/GPU performance (model training)
- Number of successful stock downloads

**Expected output:**
- Console logs showing training progress
- Test loss and RMSE metrics
- Two matplotlib visualization windows (UPS and KMB predictions)

### 2.5 Directory Structure

```
stockrecommendation/
├── main.py                 # Main application (452 lines)
├── .venv/                  # Python virtual environment
│   ├── bin/
│   ├── lib/
│   └── ...
├── .idea/                  # PyCharm IDE configuration
├── .DS_Store              # macOS metadata
└── KNOWLEDGE_BASE.md      # This documentation file
```

---

## 3. API & FUNCTION REFERENCE

### 3.1 Data Acquisition Functions

#### 3.1.1 `download_stock_data()`

**Location:** Line 31-34

**Purpose:** Downloads historical stock data from Yahoo Finance

**Signature:**
```python
def download_stock_data(tickers, start_date, end_date)
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tickers` | list[str] | Yes | List of stock ticker symbols (e.g., ['AAPL', 'MSFT']) |
| `start_date` | str | Yes | Start date in format 'YYYY-MM-DD' |
| `end_date` | str | Yes | End date in format 'YYYY-MM-DD' |

**Returns:**
- **Type:** `dict[str, pd.DataFrame]`
- **Structure:** `{ticker_symbol: DataFrame_with_OHLCV_data}`
- **DataFrame columns:** Open, High, Low, Close, Volume

**Example Usage:**
```python
tickers = ['AAPL', 'MSFT', 'GOOGL']
data = download_stock_data(tickers, '2020-01-01', '2023-12-31')
# Returns: {'AAPL': DataFrame, 'MSFT': DataFrame, 'GOOGL': DataFrame}
```

**Error Handling:**
- Network timeouts: Ticker skipped, warning printed
- Delisted stocks: Ticker skipped, YFTzMissingError caught
- Invalid tickers: Empty DataFrame returned

**Actual Call in Code (Line 48):**
```python
data = download_stock_data(tickers, '2020-01-01', '2025-12-31')
```

---

### 3.2 Technical Indicator Functions

#### 3.2.1 `calculate_rsi()`

**Location:** Line 52-58

**Purpose:** Calculates Relative Strength Index (RSI) momentum indicator

**Signature:**
```python
def calculate_rsi(df, window=14)
```

**Parameters:**
| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `df` | pd.DataFrame | - | Yes | DataFrame containing 'Close' column |
| `window` | int | 14 | No | Number of periods for RSI calculation |

**Returns:**
- **Type:** `pd.Series`
- **Range:** 0 to 100
- **Interpretation:**
  - RSI > 70: Overbought
  - RSI < 30: Oversold

**Algorithm:**
1. Calculate price deltas: `delta = Close.diff()`
2. Separate gains and losses
3. Calculate rolling average of gains and losses over window
4. Compute Relative Strength: `RS = avg_gain / avg_loss`
5. Calculate RSI: `100 - (100 / (1 + RS))`

**Example:**
```python
rsi = calculate_rsi(df, window=14)
# Returns Series with RSI values
```

**NaN Handling:** First 14 rows will be NaN (window period)

---

#### 3.2.2 `calculate_momentum()`

**Location:** Line 60-61

**Purpose:** Calculates price momentum (rate of change)

**Signature:**
```python
def calculate_momentum(df, window=10)
```

**Parameters:**
| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `df` | pd.DataFrame | - | Yes | DataFrame containing 'Close' column |
| `window` | int | 10 | No | Number of periods for momentum calculation |

**Returns:**
- **Type:** `pd.Series`
- **Unit:** Price units (same as Close price)
- **Interpretation:** Positive = upward momentum, Negative = downward momentum

**Algorithm:**
```python
return df['Close'].diff(window)
# Current Close - Close from 'window' days ago
```

**Example:**
```python
momentum = calculate_momentum(df, window=10)
# If Close today = $150, Close 10 days ago = $140, momentum = $10
```

---

#### 3.2.3 `calculate_bollinger_bands()`

**Location:** Line 63-68

**Purpose:** Calculates Bollinger Bands for volatility measurement

**Signature:**
```python
def calculate_bollinger_bands(df, window=20)
```

**Parameters:**
| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `df` | pd.DataFrame | - | Yes | DataFrame containing 'Close' column |
| `window` | int | 20 | No | Number of periods for moving average |

**Returns:**
- **Type:** `tuple[pd.Series, pd.Series]`
- **Structure:** `(upper_band, lower_band)`

**Algorithm:**
1. Calculate 20-day moving average: `MA = Close.rolling(20).mean()`
2. Calculate 20-day standard deviation: `σ = Close.rolling(20).std()`
3. Upper band: `MA + (2 × σ)`
4. Lower band: `MA - (2 × σ)`

**Statistical Properties:**
- Approximately 95% of price action occurs within the bands (2σ rule)

**Example:**
```python
upper, lower = calculate_bollinger_bands(df, window=20)
# upper: Series of upper band values
# lower: Series of lower band values
```

---

#### 3.2.4 `calculate_features()`

**Location:** Line 70-75

**Purpose:** Orchestrates calculation of all technical indicators

**Signature:**
```python
def calculate_features(df)
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `df` | pd.DataFrame | Yes | DataFrame with OHLCV data |

**Returns:**
- **Type:** `pd.DataFrame`
- **Modified in-place:** Yes (adds new columns to input df)

**New Columns Added:**
| Column | Function | Default Window |
|--------|----------|----------------|
| `RSI` | calculate_rsi() | 14 days |
| `Momentum` | calculate_momentum() | 10 days |
| `Moving_Average` | Rolling mean | 20 days |
| `Bollinger_Upper` | calculate_bollinger_bands() | 20 days |
| `Bollinger_Lower` | calculate_bollinger_bands() | 20 days |

**Example:**
```python
df = calculate_features(df)
# df now has 5 additional columns: RSI, Momentum, Moving_Average,
# Bollinger_Upper, Bollinger_Lower
```

**Code Execution (Line 77-78):**
```python
for ticker in tickers:
    data[ticker] = calculate_features(data[ticker])
```

---

### 3.3 Similarity Analysis Functions

#### 3.3.1 `find_similar_stocks()`

**Location:** Line 80-101

**Purpose:** Identifies stocks with similar price behavior using feature-based distance

**Signature:**
```python
def find_similar_stocks(data, base_ticker)
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | dict[str, pd.DataFrame] | Yes | Dictionary mapping tickers to DataFrames with features |
| `base_ticker` | str | Yes | Reference ticker to find similarities to (e.g., 'AMZN') |

**Returns:**
- **Type:** `list[str]`
- **Length:** 10 (top 10 most similar stocks)
- **Sorted:** By similarity (most similar first)

**Algorithm:**
1. Extract features from base ticker: `['RSI', 'Momentum', 'Moving_Average', 'Bollinger_Upper', 'Bollinger_Lower', 'Close']`
2. Drop NaN values from base features
3. For each other ticker:
   - Extract same features
   - Calculate pairwise Euclidean distance: `sklearn.metrics.pairwise.pairwise_distances()`
   - Compute mean distance across all time points
4. Sort tickers by distance (ascending)
5. Return top 10

**Distance Metric:**
- **Euclidean distance** in 6-dimensional feature space
- Lower distance = higher similarity

**Error Handling:**
- Base ticker has no valid features → Returns empty list, prints warning
- Other ticker has no valid features → Skips that ticker, prints warning

**Example Usage:**
```python
similar = find_similar_stocks(data, 'AMZN')
# Returns: ['PEP', 'TXN', 'TM', 'PNC', 'UPS', 'QCOM', 'KMB', 'ADI', 'TGT', 'CVX']
```

**Actual Call (Line 103):**
```python
similar_stocks = find_similar_stocks(data, 'AMZN')
```

---

### 3.4 Data Preprocessing Functions

#### 3.4.1 `create_sequences()`

**Location:** Line 175-182

**Purpose:** Converts time series data into sliding window sequences for LSTM input

**Signature:**
```python
def create_sequences(data, seq_length)
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | np.ndarray | Yes | 2D array of scaled stock data [time_steps, features] |
| `seq_length` | int | Yes | Number of time steps per sequence (lookback window) |

**Returns:**
- **Type:** `tuple[np.ndarray, np.ndarray]`
- **Structure:** `(X, y)` where:
  - `X`: 3D array `[num_sequences, seq_length, num_features]` (input sequences)
  - `y`: 2D array `[num_sequences, num_features]` (target values)

**Algorithm:**
```python
for i in range(len(data) - seq_length):
    x = data[i:i+seq_length]        # 8 days of historical data
    y = data[i+seq_length]          # Next day's values (prediction target)
    xs.append(x)
    ys.append(y)
```

**Example:**
```python
# Input: scaled_data shape (1000, 10) - 1000 days, 10 stocks
seq_length = 8
X, y = create_sequences(scaled_data, seq_length)
# X shape: (992, 8, 10) - 992 sequences, 8-day lookback, 10 features
# y shape: (992, 10) - 992 target values (next day prediction)
```

**Actual Call (Line 184):**
```python
seq_length = 8
X, y = create_sequences(scaled_data, seq_length)
```

**Why 8-day sequences?**
- Hardcoded parameter representing approximately 1.5 trading weeks
- Chosen empirically (not theoretically justified in code)

---

### 3.5 Neural Network Classes

#### 3.5.1 `LSTM` Class

**Location:** Line 197-220

**Purpose:** Custom LSTM neural network for stock price prediction

**Class Definition:**
```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size)
    def forward(self, x)
```

**Constructor Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input_size` | int | Yes | Number of input features (1 or 10 stocks) |
| `hidden_size` | int | Yes | Number of hidden units per LSTM layer (typically 100) |
| `num_layers` | int | Yes | Number of LSTM layers (always 1 in practice, architecture uses 4 separate layers) |
| `output_size` | int | Yes | Number of output features (1 or 10 stocks) |

**Architecture:**
```
Input [batch_size, seq_length, input_size]
    ↓
LSTM Layer 1 (100 units, dropout=0.4)
    ↓
LSTM Layer 2 (100 units, dropout=0.4)
    ↓
LSTM Layer 3 (100 units, dropout=0.4)
    ↓
LSTM Layer 4 (100 units, dropout=0.4)
    ↓
Fully Connected Layer [hidden_size → output_size]
    ↓
Output [batch_size, output_size]
```

**LSTM Layers:**
- `lstm1`, `lstm2`, `lstm3`, `lstm4`: Each is `nn.LSTM` with:
  - `num_layers=1` (single layer)
  - `batch_first=True` (input shape: [batch, seq, features])
  - `dropout=0.4` (40% dropout - WARNING: ineffective for single-layer LSTM)

**Forward Pass:**
```python
def forward(self, x):
    h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)  # Initial hidden state
    c0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)  # Initial cell state
    out, _ = self.lstm1(x, (h0, c0))
    out, _ = self.lstm2(out, (h0, c0))
    out, _ = self.lstm3(out, (h0, c0))
    out, _ = self.lstm4(out, (h0, c0))
    out = self.fc(out[:, -1, :])  # Take only last time step output
    return out
```

**Why Separate LSTM Layers?**
- Allows selective freezing for transfer learning
- Layers 1-3 frozen, Layer 4 + FC layer fine-tuned on recent data

**Known Issues:**
- Dropout warning: `dropout=0.4` with `num_layers=1` has no effect
- Should either use `num_layers=4` or remove dropout parameter

**Example Instantiation:**
```python
# Multi-feature model
model = LSTM(input_size=10, hidden_size=100, num_layers=1, output_size=10)

# Single-feature model
model_1 = LSTM(input_size=1, hidden_size=100, num_layers=1, output_size=1)
```

---

## 4. DATA STRUCTURES

### 4.1 Input Data Structure

**Stock Data Dictionary (after download):**
```python
data = {
    'AAPL': pd.DataFrame({
        'Open': [...],
        'High': [...],
        'Low': [...],
        'Close': [...],
        'Volume': [...]
    }),
    'MSFT': pd.DataFrame(...),
    # ... 98 more stocks
}
```

**Index:** DatetimeIndex with trading dates (2020-01-01 to 2025-12-31)

### 4.2 Feature-Enriched Data Structure

**After `calculate_features()`:**
```python
data['AAPL'] columns:
- Open
- High
- Low
- Close
- Volume
- RSI                 # Added
- Momentum            # Added
- Moving_Average      # Added
- Bollinger_Upper     # Added
- Bollinger_Lower     # Added
```

### 4.3 Training Data Tensors

**After preprocessing:**
```python
X_train: torch.Tensor        # Shape: [n_samples-5, 8, 10]
y_train: torch.Tensor        # Shape: [n_samples-5, 10]
X_test: torch.Tensor         # Shape: [5, 8, 10]
y_test: torch.Tensor         # Shape: [5, 10]

# Transfer learning splits
X_train_1: torch.Tensor      # Shape: [n_samples-20, 8, 10]
y_train_1: torch.Tensor      # Shape: [n_samples-20, 10]
X_train_2: torch.Tensor      # Shape: [15, 8, 10]  # Days -20 to -5
y_train_2: torch.Tensor      # Shape: [15, 10]
```

**Dimensions:**
- First dimension: Number of sequences
- Second dimension: Sequence length (8 days)
- Third dimension: Number of features (10 stocks or 1 stock)

### 4.4 Prediction Output Structure

**Model Predictions:**
```python
predicted_prices_0: np.ndarray    # Shape: [5, 10] - Model 0 predictions
predicted_prices_0f: np.ndarray   # Shape: [5, 10] - Model 0f predictions
predicted_prices_1: np.ndarray    # Shape: [5, 10] - Model 1 predictions
predicted_prices_1f: np.ndarray   # Shape: [5, 10] - Model 1f predictions
```

**Interpretation:**
- Rows: 5 trading days (predictions)
- Columns: 10 similar stocks
- Values: Predicted closing prices in original scale (USD)

---

## 5. MODEL ARCHITECTURE

### 5.1 Model Variants

The system trains **4 different LSTM models** with distinct configurations:

| Model | Input Features | Training Strategy | Epochs | Learning Rate |
|-------|----------------|-------------------|--------|---------------|
| **Model 0** | 10 (all stocks) | Standard | 100 | 0.001 |
| **Model 0f** | 10 (all stocks) | Transfer Learning | 100 + 100 | 0.001 |
| **Model 1** | 1 (per stock) | Per-Stock | 10 × 10 stocks | 0.002 |
| **Model 1f** | 1 (per stock) | Transfer Learning | 10 + 10 × 10 stocks | 0.002 |

### 5.2 Model 0: Multi-Feature Standard Training

**Location:** Line 226-263

**Configuration:**
```python
input_size = len(tickers)      # 10 stocks
hidden_size = 100
num_layers = 1                 # (4 separate layers defined)
output_size = len(tickers)     # 10 stocks
epochs = 100
learning_rate = 0.001
```

**Training Data:** `X_train` (all sequences except last 5)

**Loss Function:** MSELoss (Mean Squared Error)

**Optimizer:** Adam with StepLR scheduler
- `step_size=50`: Learning rate decay every 50 epochs
- `gamma=0.1`: Multiply learning rate by 0.1

**Output:** `predicted_prices_0` - 5-day forecast for 10 stocks

### 5.3 Model 0f: Multi-Feature Transfer Learning

**Location:** Line 265-322

**Training Strategy:**

**Phase 1: Pre-training (Epochs 1-100)**
- Train on `X_train_1` (all data except last 20 days)
- All layers trainable
- Learning rate: 0.001

**Phase 2: Layer Freezing (Line 283-289)**
```python
for param in model.lstm1.parameters():
    param.requires_grad = False
for param in model.lstm2.parameters():
    param.requires_grad = False
for param in model.lstm3.parameters():
    param.requires_grad = False
# Only lstm4 and fc layers remain trainable
```

**Phase 3: Fine-tuning (Epochs 101-200)**
- Train on `X_train_2` (days -20 to -5, most recent data)
- Only LSTM layer 4 and FC layer trainable
- Learning rate: 0.001 (with continued StepLR decay)

**Hypothesis:** Recent market data should guide final predictions, while frozen layers retain long-term patterns

**Output:** `predicted_prices_0f`

### 5.4 Model 1: Single-Feature Per-Stock Training

**Location:** Line 324-363

**Configuration:**
```python
input_size = 1                 # Single stock
hidden_size = 100
num_layers = 1
output_size = 1                # Single stock prediction
epochs = 10
learning_rate = 0.002          # Higher LR (faster training for simpler problem)
```

**Training Loop:**
```python
for epoch in range(10):
    for i in range(10):  # Iterate through each stock
        X_train_i = X_train[:, :, i:i+1]  # Extract single stock's data
        y_train_i = y_train[:, i:i+1]
        # Train on this stock's data
```

**Key Difference:** Model trains separately on each stock, learning individual patterns rather than cross-stock relationships

**Output Assembly:**
```python
my_outputs_1 = torch.zeros(5, 10)  # Pre-allocate tensor
for i in range(10):
    X_test_i = X_test[:, :, i:i+1]
    prediction = model_1(X_test_i)
    my_outputs_1[:, i:i+1] = prediction
```

**Output:** `predicted_prices_1`

### 5.5 Model 1f: Single-Feature Transfer Learning

**Location:** Line 365-424

**Training Strategy:**

**Phase 1: Pre-training (10 epochs)**
- Train on `X_train_1` per stock (all data except last 20 days)

**Phase 2: Freeze first 3 LSTM layers (Line 388-394)**

**Phase 3: Fine-tuning (10 epochs)**
- Train on `X_train_2` per stock (days -20 to -5)
- Simultaneously evaluate on test set and update predictions

**Unique Feature:** Predictions updated **during** fine-tuning
```python
for epoch in range(10):
    for i in range(10):
        # Train on X_train_2
        outputs = model_1(X_train_2[:, :, i:i+1])
        loss.backward()
        optimizer.step()

        # Immediately evaluate on test set
        model_1.eval()
        my_outputs_1f[:, i:i+1] = model_1(X_test[:, :, i:i+1])
```

**Output:** `predicted_prices_1f`

---

## 6. TRAINING PIPELINE

### 6.1 Data Acquisition

**Code:** Line 36-49

**Stock Universe:** 100 stocks across sectors:
- Technology: AAPL, MSFT, GOOGL, NVDA, AMD, etc.
- Finance: JPM, V, PYPL, WFC, GS, etc.
- Consumer: AMZN, WMT, COST, NKE, SBUX, etc.
- Healthcare: PFE, MRNA, ABBV, GILD, etc.
- Energy: XOM, CVX, COP, etc.
- International: BABA, JD, NIO, NVS, TM, etc.

**Date Range:** 2020-01-01 to 2025-12-31

**Expected Failures:**
- SQ (Square/Block): Delisted or missing timezone
- JPM: Occasional timeout errors

### 6.2 Feature Calculation

**Code:** Line 77-78

**Process:**
1. For each of 100 stocks:
   - Calculate RSI (14-day rolling)
   - Calculate Momentum (10-day difference)
   - Calculate Moving Average (20-day rolling mean)
   - Calculate Bollinger Bands (20-day ± 2σ)

**Result:** Each DataFrame has 10 columns (5 original + 5 calculated)

### 6.3 Similarity Filtering

**Code:** Line 103-161

**Steps:**
1. Calculate pairwise distances between AMZN and all other stocks
2. Select top 10 most similar: `['PEP', 'TXN', 'TM', 'PNC', 'UPS', 'QCOM', 'KMB', 'ADI', 'TGT', 'CVX']`
3. Filter data to 2020-01-01 to 2023-01-01 (3 years)
4. Extract only 'Close' prices for these 10 stocks
5. Align indices (handle missing dates)
6. Create DataFrame with 10 columns (one per stock)

**Output:** `df` with shape approximately (756, 10) - 756 trading days, 10 stocks

### 6.4 Scaling & Sequencing

**Code:** Line 168-194

**Scaling:**
```python
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.dropna())
# All values normalized to [0, 1] range
```

**Sequencing:**
```python
seq_length = 8
X, y = create_sequences(scaled_data, seq_length)
# X shape: (748, 8, 10)
# y shape: (748, 10)
```

**Train/Test Split:**
```python
X_train = X[:-5]      # All except last 5
y_train = y[:-5]
X_test = X[-5:]       # Last 5 sequences
y_test = y[-5:]
```

**Transfer Learning Splits:**
```python
X_train_1 = X[:-20]   # All except last 20
y_train_1 = y[:-20]
X_train_2 = X[-20:-5] # Days -20 to -5 (15 sequences)
y_train_2 = y[-20:-5]
```

### 6.5 Training Execution

**Model 0 Training Loop (Line 232-242):**
```python
for epoch in range(100):
    model.train()
    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()
    scheduler.step()
```

**Typical Loss Progression:**
```
Epoch [10/100], Loss: 0.2211
Epoch [50/100], Loss: 0.0213
Epoch [100/100], Loss: 0.0176
```

**Model 1 Training Loop (Line 332-345):**
```python
for epoch in range(10):
    for i in range(10):  # Loop through stocks
        X_train_i = X_train[:, :, i:i+1]
        y_train_i = y_train[:, i:i+1]
        outputs = model_1(X_train_i.to(device))
        loss.backward()
        optimizer.step()
```

### 6.6 Evaluation & Prediction

**Evaluation (Line 244-258):**
```python
model.eval()
outputs = model(X_test.to(device))
test_loss = criterion(outputs, y_test.to(device))
test_rmse = math.sqrt(test_loss.item())
```

**Inverse Scaling (Line 261-262):**
```python
scaled_predicted_prices = outputs.cpu().detach().numpy()
predicted_prices_0 = scaler.inverse_transform(scaled_predicted_prices)
# Convert from [0,1] scale back to actual prices
```

---

## 7. COMMON TROUBLESHOOTING

### 7.1 Installation Issues

#### **Problem:** `ModuleNotFoundError: No module named 'torch'`

**Cause:** PyTorch not installed or virtual environment not activated

**Solution:**
```bash
# Activate virtual environment first
source .venv/bin/activate

# Install PyTorch
pip install torch==2.9.1
```

---

#### **Problem:** `ImportError: cannot import name 'xxx' from 'sklearn'`

**Cause:** scikit-learn version mismatch

**Solution:**
```bash
pip install --upgrade scikit-learn==1.8.0
```

---

### 7.2 Runtime Errors

#### **Problem:** `YFTzMissingError('possibly delisted; no timezone found')`

**Ticker:** SQ (Square/Block)

**Cause:** Stock has been delisted or ticker symbol changed

**Impact:** Non-critical - system continues with remaining 99 stocks

**Solution:** No action required - warning can be ignored

---

#### **Problem:** `Timeout('Failed to perform, curl: (28) Operation timed out...')`

**Ticker:** JPM or others

**Cause:** Network connection timeout or Yahoo Finance API rate limiting

**Impact:** Non-critical - system continues with available stocks

**Solution:**
1. Check internet connection
2. Retry execution
3. If persistent, remove problematic ticker from list (line 36-47)

---

#### **Problem:** `RuntimeError: MPS backend out of memory`

**Cause:** Apple Silicon GPU has insufficient memory for batch size

**Solution:**
```python
# Option 1: Force CPU usage
device = torch.device("cpu")

# Option 2: Reduce hidden_size (line 226, 325)
hidden_size = 50  # Instead of 100
```

---

#### **Problem:** `KeyError: 'Close'` when processing similar stocks

**Cause:** Data structure mismatch or missing stock data

**Location:** Line 108-133

**Solution:** Code already has fallback logic - if error persists, check:
```python
print(data[ticker].columns)  # Verify column names
print(data[ticker].head())   # Check data structure
```

---

### 7.3 PyTorch Warnings

#### **Warning:** `dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.4 and num_layers=1`

**Cause:** LSTM instantiated with `num_layers=1` but `dropout=0.4`

**Location:** Line 202-205 (lstm1, lstm2, lstm3, lstm4 definitions)

**Impact:** Dropout is **not applied** - warning is informational

**Solution (if you want functional dropout):**
```python
# Option 1: Remove dropout parameter
self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

# Option 2: Use num_layers=2 or more (changes architecture)
self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.4)
```

**Why current design exists:** Separate layers allow selective freezing for transfer learning

---

### 7.4 Data Issues

#### **Problem:** `No valid stock data available to create DataFrame`

**Cause:** All similar stocks failed to download or have empty data

**Solution:**
1. Check internet connection
2. Verify ticker symbols are valid
3. Check date range (2020-01-01 to 2023-01-01 must have data)

---

#### **Problem:** `UserWarning: Ticker 'XXX' has no valid features after dropping NaNs`

**Cause:** Stock has insufficient historical data or too many missing values

**Impact:** Stock excluded from similarity calculation

**Solution:** No action required - system continues with available stocks

---

#### **Problem:** Predictions are all similar values (no variation)

**Cause:** Model underfitting or data scaling issue

**Debugging:**
```python
# Check scaled data range
print(f"Min: {scaled_data.min()}, Max: {scaled_data.max()}")  # Should be 0 and 1

# Check model output before inverse scaling
print(outputs.min(), outputs.max())  # Should be in [0, 1] range
```

---

### 7.5 Visualization Issues

#### **Problem:** `UserWarning: FigureCanvasAgg is non-interactive`

**Cause:** Matplotlib backend not configured for display

**Solution:** Already handled in code (line 6-7):
```python
matplotlib.use('TkAgg')
```

If still not working:
```bash
# macOS
brew install python-tk

# Linux
sudo apt-get install python3-tk
```

---

#### **Problem:** Plots don't appear / script hangs

**Cause:** `plt.show()` is blocking execution

**Solution:**
1. Use non-blocking mode:
```python
plt.show(block=False)
plt.pause(5)  # Display for 5 seconds
```

2. Save plots instead:
```python
plt.savefig('ups_predictions.png')
plt.close()
```

---

### 7.6 Performance Issues

#### **Problem:** Training takes extremely long (>30 minutes)

**Cause:** Running on CPU instead of GPU

**Diagnosis:**
```python
print(f"Using device: {device}")  # Should show "mps" or "cuda"
```

**Solution:**
```bash
# Check GPU availability
python -c "import torch; print(torch.backends.mps.is_available())"  # macOS
python -c "import torch; print(torch.cuda.is_available())"          # NVIDIA

# If False, GPU drivers not installed
```

---

#### **Problem:** `NaN` loss during training

**Cause:** Gradient explosion or learning rate too high

**Solution:**
```python
# Reduce learning rate (line 228, 327, 368)
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Instead of 0.001

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 7.7 Memory Issues

#### **Problem:** `MemoryError` or system freezes

**Cause:** Loading too much data into RAM

**Solution:**
1. Reduce stock universe:
```python
tickers = tickers[:50]  # Use only 50 stocks instead of 100
```

2. Reduce sequence length:
```python
seq_length = 5  # Instead of 8
```

3. Use smaller hidden size:
```python
hidden_size = 50  # Instead of 100
```

---

## 8. CONFIGURATION REFERENCE

### 8.1 Hardcoded Parameters

All configuration is hardcoded in `main.py`. To modify:

#### **Stock Universe (Line 36-47)**
```python
tickers = [
    'AAPL', 'MSFT', ...  # Add/remove tickers here
]
```

#### **Date Range (Line 48)**
```python
data = download_stock_data(tickers, '2020-01-01', '2025-12-31')
```

#### **Training Data Date Range (Line 117, 122)**
```python
close_prices = data[ticker]['Close'].loc['2020-01-01':'2023-01-01']
```

#### **Technical Indicator Windows**
```python
# RSI (Line 52)
def calculate_rsi(df, window=14)

# Momentum (Line 60)
def calculate_momentum(df, window=10)

# Bollinger Bands (Line 63)
def calculate_bollinger_bands(df, window=20)

# Moving Average (Line 73)
df['Moving_Average'] = df['Close'].rolling(window=20).mean()
```

#### **Similarity Analysis (Line 81)**
```python
features = ['RSI', 'Momentum', 'Moving_Average', 'Bollinger_Upper', 'Bollinger_Lower', 'Close']
similar_stocks = sorted(similarities, key=similarities.get)[:10]  # Top 10
```

#### **Sequence Length (Line 183)**
```python
seq_length = 8  # Number of days for LSTM lookback
```

#### **Model Hyperparameters**

**Model 0/0f (Line 226-229):**
```python
model = LSTM(input_size=10, hidden_size=100, num_layers=1, output_size=10)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
num_epochs = 100
```

**Model 1/1f (Line 325-328):**
```python
model_1 = LSTM(input_size=1, hidden_size=100, num_layers=1, output_size=1)
optimizer = optim.Adam(model_1.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
num_epochs = 10
```

#### **LSTM Architecture (Line 202-208)**
```python
self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.4)
self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=0.4)
self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=0.4)
self.lstm4 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=0.4)
self.fc = nn.Linear(hidden_size, output_size)
```

#### **Train/Test Split (Line 188-194)**
```python
X_train, y_train = X[:-5], y[:-5]       # All except last 5 days
X_test, y_test = X[-5:], y[-5:]         # Last 5 days (test set)

X_train_1, y_train_1 = X[:-20], y[:-20] # All except last 20 days
X_train_2, y_train_2 = X[-20:-5], y[-20:-5]  # Days -20 to -5
```

#### **Visualization (Line 427-445)**
```python
# Stocks to visualize
plt.plot(df['UPS'][-5:], label='Actual UPS Prices')  # Line 428
plt.plot(df['KMB'][-5:], label='Actual KMB Prices')  # Line 438

# Y-axis minimum
plt.ylim(bottom=100)  # Line 434, 444
```

### 8.2 Modifiable Configuration Guide

**To change base ticker for similarity:**
```python
similar_stocks = find_similar_stocks(data, 'AAPL')  # Instead of 'AMZN'
```

**To predict more days:**
```python
X_test, y_test = X[-10:], y[-10:]  # Predict last 10 days instead of 5
```

**To train longer:**
```python
num_epochs = 200  # Instead of 100
```

**To change learning rate:**
```python
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower LR
```

---

## 9. EXECUTION FLOW

### 9.1 Complete Execution Sequence

```
START
  ↓
[1] Device Selection (Line 21-28)
  ├─ Check MPS (Apple Silicon)
  ├─ Check CUDA (NVIDIA GPU)
  └─ Fallback to CPU
  ↓
[2] Download Stock Data (Line 48)
  ├─ Download 100 stocks (2020-2025)
  ├─ Handle failures (SQ, JPM)
  └─ Store in data dictionary
  ↓
[3] Calculate Features (Line 77-78)
  ├─ RSI (14-day)
  ├─ Momentum (10-day)
  ├─ Moving Average (20-day)
  └─ Bollinger Bands (20-day, 2σ)
  ↓
[4] Find Similar Stocks (Line 103)
  ├─ Base ticker: AMZN
  ├─ Calculate Euclidean distances
  └─ Select top 10
  ↓
[5] Filter & Align Data (Line 108-161)
  ├─ Extract Close prices (2020-2023)
  ├─ Align indices
  └─ Create df (10 stocks)
  ↓
[6] Scale & Sequence (Line 168-194)
  ├─ MinMaxScaler [0,1]
  ├─ Create 8-day sequences
  └─ Split train/test
  ↓
[7] Train Model 0 (Line 226-263)
  ├─ Multi-feature LSTM
  ├─ 100 epochs
  ├─ Predict & evaluate
  └─ Output: predicted_prices_0
  ↓
[8] Train Model 0f (Line 265-322)
  ├─ Pre-train (100 epochs)
  ├─ Freeze layers 1-3
  ├─ Fine-tune (100 epochs)
  ├─ Predict & evaluate
  └─ Output: predicted_prices_0f
  ↓
[9] Train Model 1 (Line 324-363)
  ├─ Single-feature LSTM
  ├─ 10 epochs × 10 stocks
  ├─ Predict & evaluate
  └─ Output: predicted_prices_1
  ↓
[10] Train Model 1f (Line 365-424)
  ├─ Pre-train (10 epochs)
  ├─ Freeze layers 1-3
  ├─ Fine-tune (10 epochs)
  ├─ Predict & evaluate
  └─ Output: predicted_prices_1f
  ↓
[11] Visualize Results (Line 427-445)
  ├─ Plot UPS predictions (4 models)
  ├─ Plot KMB predictions (4 models)
  └─ Display matplotlib windows
  ↓
END
```

### 9.2 Expected Console Output

```
Using device: mps

Price            Open       High        Low      Close     Volume
Date
2020-01-02  71.476585  72.528566  71.223244  72.468246  135480400
...

Warning: Ticker 'SQ' has no valid features after dropping NaNs.
Warning: Ticker 'JPM' has no valid features after dropping NaNs.

Similar stocks: ['PEP', 'TXN', 'TM', 'PNC', 'UPS', 'QCOM', 'KMB', 'ADI', 'TGT', 'CVX']

[[0.31973646 0.30352003 ...]]

Epoch [10/100], Loss: 0.2211
Epoch [20/100], Loss: 0.0402
...
Epoch [100/100], Loss: 0.0176

Test Loss: 0.0471
Test RMSE: 0.2170

[[139.7486 154.83862 ...]]

[Additional model training output...]

[Matplotlib windows appear with plots]
```

### 9.3 Execution Time Breakdown

| Phase | Approximate Time | Bottleneck |
|-------|------------------|------------|
| Device selection | <1 second | - |
| Data download | 30-60 seconds | Network speed |
| Feature calculation | 5-10 seconds | CPU |
| Similarity analysis | 2-5 seconds | CPU |
| Data filtering | 1-2 seconds | - |
| Scaling & sequencing | 1 second | - |
| Model 0 training | 30-60 seconds | GPU/CPU |
| Model 0f training | 60-120 seconds | GPU/CPU |
| Model 1 training | 10-20 seconds | GPU/CPU |
| Model 1f training | 20-40 seconds | GPU/CPU |
| Visualization | 2-5 seconds | Display |
| **TOTAL** | **3-8 minutes** | GPU vs CPU |

**GPU Acceleration Impact:**
- Apple Silicon (MPS): ~3-5 minutes
- CUDA GPU: ~3-5 minutes
- CPU only: ~10-20 minutes

---

## 10. PERFORMANCE BENCHMARKS

### 10.1 Expected Model Performance

Based on typical execution:

| Model | Test Loss (MSE) | Test RMSE | Relative Performance |
|-------|----------------|-----------|----------------------|
| **Model 0** | 0.0471 | 0.2170 | Baseline |
| **Model 0f** | 0.0361 | 0.1900 | 12% better than Model 0 |
| **Model 1** | 0.0030 | 0.0551 | 75% better than Model 0 |
| **Model 1f** | 0.0029 | 0.0542 | Best (76% better than Model 0) |

**Key Finding:** Single-feature per-stock models (Model 1/1f) significantly outperform multi-feature models (Model 0/0f)

**Hypothesis:** Cross-stock correlations introduce noise; individual stock patterns are more predictable

### 10.2 Training Loss Convergence

**Model 0 (Multi-feature, 100 epochs):**
```
Epoch [10/100], Loss: 0.2211  (Starting loss)
Epoch [50/100], Loss: 0.0213  (Rapid improvement)
Epoch [100/100], Loss: 0.0176 (Converged)
```

**Model 1 (Single-feature, 10 epochs):**
```
Epoch [2/10], Loss: 0.0878   (Starting loss)
Epoch [6/10], Loss: 0.0039   (Rapid convergence)
Epoch [10/10], Loss: 0.0024  (Final loss)
```

**Observation:** Model 1 converges faster (10 epochs) than Model 0 (100 epochs) despite simpler architecture

### 10.3 Sample Predictions

**UPS Stock - Last 5 Days (Model 1f):**
```
Day 1: $158.17
Day 2: $157.91
Day 3: $157.66
Day 4: $157.78
Day 5: $157.92
```

**Prediction Pattern:** Slight downward trend followed by small recovery (common LSTM behavior)

### 10.4 System Resource Usage

**Memory Consumption:**
- Data loading: ~500 MB
- Model training (GPU): ~1-2 GB GPU RAM
- Model training (CPU): ~2-3 GB system RAM
- Peak usage: ~3-4 GB total

**CPU Utilization:**
- Data download/processing: 10-30% (single core)
- GPU training: 5-10% CPU, 60-90% GPU
- CPU training: 80-100% (all cores)

**Disk Usage:**
- Virtual environment: ~1.5 GB
- Code: <1 MB
- No persistent model files (not saved)

---

## APPENDIX A: STOCK TICKER LIST

Complete list of 100 stocks used in system (Line 36-47):

**Technology (18):**
AAPL, MSFT, GOOGL, NVDA, AMD, INTC, CSCO, ADBE, ORCL, CRM, IBM, TXN, LRCX, MU, AMAT, ADI, SPOT, PLTR

**Consumer & Retail (15):**
AMZN, WMT, COST, NKE, SBUX, MCD, TGT, LOW, KHC, MDLZ, EL, BKNG, DIS, NFLX, META

**Finance (11):**
JPM, V, PYPL, WFC, GS, MS, SCHW, C, USB, PNC, SPGI

**Healthcare & Pharma (8):**
PFE, MRNA, ABBV, BMY, GILD, MRK, TMO, ISRG, BSX

**Energy (7):**
XOM, CVX, COP, VLO, OXY, PSX, NEE

**Industrial & Manufacturing (9):**
BA, CAT, LMT, GE, UPS, FDX, DE, NOC, DD

**Consumer Goods (6):**
PEP, KO, MO, PM, KMB, CL, DOW

**Utilities (4):**
T, VZ, EXC, SRE, SO, AEP

**Automotive (2):**
F, GM, TSLA

**International (8):**
BABA, JD, NIO, NVS, TM, RIO, UL

**Technology (Growth) (5):**
SQ, ZM, SNOW, SHOP, RBLX

**Insurance (1):**
PGR

---

## APPENDIX B: Mathematical Formulas

### RSI Formula
```
Δ = Close[t] - Close[t-1]
Gain = Δ if Δ > 0, else 0
Loss = -Δ if Δ < 0, else 0

Avg_Gain = SMA(Gain, 14)
Avg_Loss = SMA(Loss, 14)

RS = Avg_Gain / Avg_Loss
RSI = 100 - (100 / (1 + RS))
```

### Momentum Formula
```
Momentum[t] = Close[t] - Close[t-10]
```

### Bollinger Bands Formula
```
MA = SMA(Close, 20)
σ = StdDev(Close, 20)

Upper_Band = MA + (2 × σ)
Lower_Band = MA - (2 × σ)
```

### Euclidean Distance Formula
```
For stocks A and B with feature vectors:
A = [RSI_A, Momentum_A, MA_A, BB_Upper_A, BB_Lower_A, Close_A]
B = [RSI_B, Momentum_B, MA_B, BB_Upper_B, BB_Lower_B, Close_B]

Distance = √(Σ(A[i] - B[i])²) for i = 0 to 5
```

### MSE Loss Formula
```
MSE = (1/n) × Σ(y_pred - y_actual)²
RMSE = √MSE
```

---

## APPENDIX C: Code Location Reference

Quick reference for finding specific functionality:

| Functionality | Line Range | Function/Class |
|--------------|------------|----------------|
| Device selection | 21-28 | Global code |
| Data download | 31-34 | `download_stock_data()` |
| Stock universe | 36-47 | `tickers` list |
| RSI calculation | 52-58 | `calculate_rsi()` |
| Momentum calculation | 60-61 | `calculate_momentum()` |
| Bollinger Bands | 63-68 | `calculate_bollinger_bands()` |
| Feature orchestration | 70-75 | `calculate_features()` |
| Similarity analysis | 80-101 | `find_similar_stocks()` |
| Data filtering | 108-161 | Inline code |
| Scaling | 168-169 | MinMaxScaler |
| Sequencing | 175-182 | `create_sequences()` |
| Train/test split | 188-194 | Inline code |
| LSTM class definition | 197-220 | `class LSTM` |
| Model 0 training | 226-263 | Inline code |
| Model 0f training | 265-322 | Inline code |
| Model 1 training | 324-363 | Inline code |
| Model 1f training | 365-424 | Inline code |
| Visualization | 427-445 | Matplotlib plots |

---

## APPENDIX D: Glossary

**AMZN:** Amazon.com Inc. ticker symbol, used as base for similarity analysis

**Bollinger Bands:** Volatility indicator with upper/lower bands at ±2σ from moving average

**Dropout:** Regularization technique randomly disabling neurons during training

**Euclidean Distance:** L2 distance metric measuring similarity between feature vectors

**Fine-tuning:** Training subset of model layers on recent data (transfer learning phase 2)

**Layer Freezing:** Disabling gradient updates for specific layers (`requires_grad=False`)

**LSTM:** Long Short-Term Memory - recurrent neural network for sequence modeling

**MPS:** Metal Performance Shaders - Apple's GPU acceleration framework

**MSE:** Mean Squared Error - loss function measuring average squared prediction error

**OHLCV:** Open, High, Low, Close, Volume - standard stock price data format

**Pre-training:** Initial training phase on full historical data (transfer learning phase 1)

**RMSE:** Root Mean Squared Error - square root of MSE, in original price units

**RSI:** Relative Strength Index - momentum oscillator measuring overbought/oversold

**Scaler:** MinMaxScaler - normalization transforming data to [0,1] range

**Sequence Length:** Number of time steps (8 days) used as LSTM input

**Similarity:** Inverse of Euclidean distance - lower distance = higher similarity

**Transfer Learning:** Training strategy with pre-training + layer freezing + fine-tuning

**yfinance:** Python library for downloading Yahoo Finance stock data

---

## REVISION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-23 | Initial KNOWLEDGE_BASE.md creation |

---

**END OF KNOWLEDGE BASE**
