"""
XGBoost direction classifier — tree-based strong baseline.

Answers the diagnostic question:
  "Is 56.9% the LSTM ceiling, or the *feature* ceiling?"

  - If XGBoost > LSTM  → architecture is the bottleneck (features have more
                          signal than the LSTM can extract).
  - If XGBoost ≈ LSTM  → features are the bottleneck (both models extract the
                          same signal; no architecture fix will help without
                          better data).
  - If XGBoost < LSTM  → temporal dynamics matter; the LSTM's sequence modelling
                          is doing real work that a flat classifier cannot replicate.

Two input strategies (controlled by use_last_step_only):

  Last-step (default):
      Each training sample → the final time-step snapshot of
      (n_features_per_stock + n_macro) features per stock.
      Technical indicators (RSI, MACD, BB…) already encode recent history
      internally, so the last snapshot captures most of the relevant signal.
      Input size: n_features_per_stock + n_macro  (e.g. 13 for 10 feat + 3 macro)

  Full-sequence flatten (use_last_step_only=False):
      Each sample → all seq_length × (n_features + n_macro) features flattened.
      Captures how indicators evolved over the lookback window.
      Input size: seq_length × (n_features_per_stock + n_macro)  (e.g. 780 for 60×13)

One XGBClassifier is trained per stock (same as Model 1/1f architecture).
The .predict() interface is identical to LSTMPredictor / SingleFeaturePredictor
so the predictor plugs into walk_forward_predict without modification.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# Default XGBoost hyperparameters — tuned for:
#   - ~700 binary training samples per stock
#   - ~13–780 features depending on input strategy
#   - Emphasis on regularisation (tree depth, min_child_weight, gamma, colsample)
_DEFAULT_XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=5,
    gamma=1.0,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)


class XGBoostPredictor:
    """
    Walk-forward compatible XGBoost binary direction classifier.

    Interface mirrors LSTMPredictor / SingleFeaturePredictor so it can be
    dropped into the walk_forward_predict loop without changes.

    Parameters
    ----------
    input_scaler : sklearn scaler, already fit on training data
    n_stocks : int
    n_features_per_stock : int   — stock-specific features (e.g. 10)
    n_macro : int                — shared macro features appended at the end (e.g. 3)
    name : str
    use_last_step_only : bool
        True  → use only the last time-step snapshot per stock + macro (default)
        False → flatten full sequence per stock + macro
    """

    def __init__(
        self,
        input_scaler,
        n_stocks: int,
        n_features_per_stock: int,
        n_macro: int,
        name: str = "XGBoost",
        use_last_step_only: bool = True,
        **xgb_kwargs,
    ):
        if not HAS_XGB:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            )
        self.input_scaler = input_scaler
        self.n_stocks = n_stocks
        self.n_features_per_stock = n_features_per_stock
        self.n_macro = n_macro
        self.name = name
        self.use_last_step_only = use_last_step_only
        self.xgb_params = {**_DEFAULT_XGB_PARAMS, **xgb_kwargs}
        self.models: dict = {}          # stock_idx → XGBClassifier
        self.val_da: Optional[float] = None   # used by EnsembleWeighted if included

    # ------------------------------------------------------------------
    # Internal feature preparation
    # ------------------------------------------------------------------

    def _scale(self, X_unscaled: np.ndarray) -> np.ndarray:
        """Standardise using the pre-fit input_scaler."""
        n, seq, k = X_unscaled.shape
        means  = self.input_scaler.mean_[:k]
        scales = self.input_scaler.scale_[:k]
        return (X_unscaled - means) / scales   # (n, seq, k)

    def _build_feature_blocks(self, X_unscaled: np.ndarray) -> list[np.ndarray]:
        """
        Return a list of 2D arrays, one per stock: shape (n, n_input_features).

        For use_last_step_only=True:
            each block = last time-step stock features + last macro features
        For use_last_step_only=False:
            each block = all time steps of stock features + all macro features,
            flattened along the time axis
        """
        scaled = self._scale(X_unscaled)     # (n, seq, k)
        n, seq, k = scaled.shape
        stock_cols = self.n_stocks * self.n_features_per_stock

        blocks = []
        for j in range(self.n_stocks):
            start = j * self.n_features_per_stock
            end   = start + self.n_features_per_stock

            if self.use_last_step_only:
                stock_feat = scaled[:, -1, start:end]          # (n, n_feat)
                macro_feat = (scaled[:, -1, stock_cols:stock_cols + self.n_macro]
                              if self.n_macro > 0 else np.empty((n, 0)))
            else:
                stock_feat = scaled[:, :, start:end].reshape(n, -1)   # (n, seq*n_feat)
                macro_feat = (scaled[:, :, stock_cols:stock_cols + self.n_macro].reshape(n, -1)
                              if self.n_macro > 0 else np.empty((n, 0)))

            blocks.append(np.concatenate([stock_feat, macro_feat], axis=1))
        return blocks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_unscaled_train: np.ndarray, y_returns_train: np.ndarray) -> None:
        """
        Train one XGBClassifier per stock.

        Parameters
        ----------
        X_unscaled_train : (T, seq_length, total_features)  — unscaled features
        y_returns_train  : (T, n_stocks)  — raw log-returns (converted to binary here)
        """
        blocks   = self._build_feature_blocks(X_unscaled_train)
        y_binary = (y_returns_train > 0).astype(int)

        for j in range(self.n_stocks):
            clf = xgb.XGBClassifier(**self.xgb_params)
            clf.fit(blocks[j], y_binary[:, j])
            self.models[j] = clf

    def predict(self, X_unscaled: np.ndarray) -> np.ndarray:
        """
        Predict direction as ±1e-4 signs.

        Parameters
        ----------
        X_unscaled : (T, seq_length, total_features)

        Returns
        -------
        np.ndarray of shape (T, n_stocks) with values ±1e-4
            +1e-4 → predict up, -1e-4 → predict down
        """
        blocks = self._build_feature_blocks(X_unscaled)
        T = X_unscaled.shape[0]
        result = np.zeros((T, self.n_stocks))
        for j in range(self.n_stocks):
            proba = self.models[j].predict_proba(blocks[j])[:, 1]  # P(up)
            result[:, j] = np.where(proba >= 0.5, 1e-4, -1e-4)
        return result

    def feature_importance(self) -> dict[int, np.ndarray]:
        """Return per-stock feature importance arrays (gain)."""
        return {
            j: clf.feature_importances_
            for j, clf in self.models.items()
        }
