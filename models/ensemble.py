"""
Ensemble predictors — variance reduction for free.

Two ensemble flavours are provided:

* ``EnsembleAverage``  — simple unweighted mean (legacy / baseline).
* ``EnsembleWeighted`` — softmax-weighted mean using each constituent's
  ``val_da`` attribute (validation-set directional accuracy set by
  ``main.py`` after training).  Falls back to equal weights if any
  constituent has ``val_da = None``.

Each constituent must produce predictions of the same shape; no scaling or
unscaling happens inside the ensemble — it operates entirely in unscaled
log-return space, the same as the LSTM predictor wrappers and the baselines.
"""
from typing import List, Optional, Sequence
import numpy as np


class EnsembleAverage:
    """
    Elementwise mean of several predictors' unscaled-return outputs.

    Example:
        >>> ens = EnsembleAverage([lstm_pred_0, lstm_pred_0f, lstm_pred_1, lstm_pred_1f])
        >>> y_hat = ens.predict(X_oos_unscaled)
    """

    def __init__(self, predictors: Sequence, name: str = "Ensemble (avg)"):
        if not predictors:
            raise ValueError("EnsembleAverage requires at least one predictor")
        self.predictors = list(predictors)
        self.name = name
        # Inherit dispatch type from constituents (all expected to match).
        types = {getattr(p, "input_type", "features") for p in self.predictors}
        if len(types) > 1:
            raise ValueError(
                f"EnsembleAverage constituents have mixed input_type: {types}"
            )
        self.input_type = next(iter(types))

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds: List[np.ndarray] = []
        ref_shape = None
        for p in self.predictors:
            out = p.predict(X)
            if ref_shape is None:
                ref_shape = out.shape
            elif out.shape != ref_shape:
                raise ValueError(
                    f"Ensemble constituent {getattr(p, 'name', type(p).__name__)} "
                    f"returned shape {out.shape}, expected {ref_shape}"
                )
            preds.append(out)
        stacked = np.stack(preds, axis=0)  # (n_models, n_samples, n_stocks)
        return stacked.mean(axis=0)


# ---------------------------------------------------------------------------


class EnsembleWeighted:
    """
    Softmax-weighted mean of several predictors using validation-set
    directional accuracy as the quality signal.

    Each predictor is expected to expose a ``val_da: float`` attribute
    (0–1 scale) set by ``main.py`` after training.  When all ``val_da``
    values are available the weight of predictor *i* is:

        weight_i = softmax( (val_da_i - mean_val_da) * temperature )

    with ``temperature=20`` so that a 2pp gap between the best and worst
    model translates to roughly a 2:1 weight ratio.  This is aggressive
    enough to favour the best model clearly while still benefiting from
    variance reduction.

    Falls back to equal weights if any constituent has ``val_da = None``.

    Example:
        >>> ens = EnsembleWeighted([pred_0, pred_0f, pred_1, pred_1f])
        >>> y_hat = ens.predict(X_oos_unscaled)
    """

    # Temperature: higher → more aggressive winner-takes-most weighting.
    # At T=50 a 5pp val_da gap gives ~60% weight to the best model;
    # at T=20 the same gap gives only ~34%.
    _TEMPERATURE: float = 50.0

    def __init__(self, predictors: Sequence, name: str = "Ensemble (weighted)"):
        if not predictors:
            raise ValueError("EnsembleWeighted requires at least one predictor")
        self.predictors = list(predictors)
        self.name = name
        types = {getattr(p, "input_type", "features") for p in self.predictors}
        if len(types) > 1:
            raise ValueError(
                f"EnsembleWeighted constituents have mixed input_type: {types}"
            )
        self.input_type = next(iter(types))

    def _weights(self) -> List[float]:
        """Compute softmax weights from val_da; fall back to uniform if missing."""
        da_values: List[Optional[float]] = [
            getattr(p, "val_da", None) for p in self.predictors
        ]
        if any(v is None for v in da_values):
            n = len(self.predictors)
            return [1.0 / n] * n

        da_arr = np.array(da_values, dtype=float)
        scores = (da_arr - da_arr.mean()) * self._TEMPERATURE
        exp_scores = np.exp(scores - scores.max())   # numerically stable softmax
        weights = exp_scores / exp_scores.sum()
        return weights.tolist()

    def predict(self, X: np.ndarray) -> np.ndarray:
        weights = self._weights()
        preds: List[np.ndarray] = []
        ref_shape = None
        for p in self.predictors:
            out = p.predict(X)
            if ref_shape is None:
                ref_shape = out.shape
            elif out.shape != ref_shape:
                raise ValueError(
                    f"EnsembleWeighted constituent "
                    f"{getattr(p, 'name', type(p).__name__)} "
                    f"returned shape {out.shape}, expected {ref_shape}"
                )
            preds.append(out)
        stacked = np.stack(preds, axis=0)       # (n_models, n_samples, n_stocks)
        w = np.array(weights)[:, None, None]    # broadcast over (samples, stocks)
        return (stacked * w).sum(axis=0)
