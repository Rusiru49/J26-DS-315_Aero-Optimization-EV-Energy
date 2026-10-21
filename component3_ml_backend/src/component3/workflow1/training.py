"""
Train and compare XGBoost, Random Forest, and an ANN (small MLP) on the
augmented (real + synthetic) dataset, 5 shape params -> Cd directly.

Score every candidate ONLY against held-out REAL CFD runs, never against
the synthetic rows (or against a random split of the augmented table) — the
synthetic rows carry no independent information beyond the real runs they
were derived from via POD, so a flexible model can fit them closely while
only learning the augmentation method's own artifacts. Held-out real
accuracy is the only trustworthy signal, and is the sole criterion used
here to pick the deployed model.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

from component3 import config
CANDIDATE_BUILDERS = {
    "xgboost": lambda: XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=config.RANDOM_STATE
    ),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=config.RANDOM_STATE
    ),
    # Scaled inputs and target: the 5 shape values span very different ranges
    # (degrees vs millimetres) and Cd is about 0.25, which an unscaled MLP
    # cannot fit.
    "ann_mlp": lambda: TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", max_iter=5000, random_state=config.RANDOM_STATE),
        ),
        transformer=StandardScaler(),
    ),
}


class FineTunedANN:
    """The ANN for augmented data, pre-trained then fine-tuned: first it learns
    the general shape -> Cd trend from the real + synthetic rows, then it keeps
    training on the REAL rows only, with a 10x smaller learning rate for a fixed
    number of passes, so the real CFD has the last word. The scalers come from
    the pre-training data and stay fixed. Settings fixed before testing
    (2026-10-21), not tuned to the results."""

    FINE_TUNE_EPOCHS = 300
    FINE_TUNE_LEARNING_RATE = 1e-4

    def fit(self, x_pre, y_pre, x_real, y_real):
        self.x_scaler = StandardScaler().fit(x_pre)
        self.y_mean, self.y_std = float(np.mean(y_pre)), float(np.std(y_pre)) or 1.0
        self.mlp = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", max_iter=5000,
                                random_state=config.RANDOM_STATE, warm_start=True)
        self.mlp.fit(self.x_scaler.transform(x_pre), (y_pre - self.y_mean) / self.y_std)
        self.mlp.set_params(max_iter=self.FINE_TUNE_EPOCHS, learning_rate_init=self.FINE_TUNE_LEARNING_RATE)
        self.mlp.fit(self.x_scaler.transform(x_real), (y_real - self.y_mean) / self.y_std)
        return self

    def predict(self, x):
        return self.mlp.predict(self.x_scaler.transform(x)) * self.y_std + self.y_mean


def real_weights(n_real, n_synthetic):
    """Sample weights that give the real rows and the synthetic rows HALF the
    say each in training (otherwise about 22 real rows vs 1,200 synthetic ones
    is 2% vs 98%)."""
    return np.concatenate([np.full(n_real, n_synthetic / max(n_real, 1)), np.ones(n_synthetic)])


def _to_xy(df):
    x = df[config.SHAPE_PARAM_NAMES].to_numpy(dtype=np.float64)
    y = df["Cd"].to_numpy(dtype=np.float64)
    return x, y


def _fit_and_score(train_df, holdout_df):
    """Train every candidate on train_df, score each on holdout_df (held-out
    REAL designs). Returns (fitted models by name, results table)."""
    x_train, y_train = _to_xy(train_df)
    x_holdout, y_holdout = _to_xy(holdout_df)
    rows, fitted = [], {}
    for name, build in CANDIDATE_BUILDERS.items():
        model = build()
        model.fit(x_train, y_train)
        fitted[name] = model
        y_pred = model.predict(x_holdout)
        rows.append({
            "model": name,
            "holdout_mae": mean_absolute_error(y_holdout, y_pred),
            "holdout_rmse": float(np.sqrt(np.mean((y_holdout - y_pred) ** 2))),
            "holdout_max_error": float(np.max(np.abs(y_holdout - y_pred))),
        })
    return fitted, pd.DataFrame(rows).sort_values("holdout_mae").reset_index(drop=True)


def train_and_select(augmented_df, holdout_df):
    """
    Train every candidate on the augmented data (real + synthetic rows) and
    score each ONLY on the held-out real designs. Returns
    (best_name, best_model, results_df).
    """
    fitted, results_df = _fit_and_score(augmented_df, holdout_df)
    best_name = results_df.iloc[0]["model"]
    return best_name, fitted[best_name], results_df


def real_only_baseline(real_df, holdout_df):
    """The same candidates trained on the real designs ONLY (no augmentation),
    scored on the same hold-out designs. This is the comparison that shows
    whether the augmentation actually helps."""
    return _fit_and_score(real_df, holdout_df)[1]


def save_model(model, name, out_dir=config.MODELS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"drag_surrogate_{name}.joblib")
    joblib.dump(model, path)
    return path


def load_model(path):
    return joblib.load(path)


def predict_cd(model, params):
    """Convenience: shape params dict -> predicted Cd from the deployed
    surrogate (NOT the internal POD/GP path — this is the final, deployed
    model used by the optimiser and the API)."""
    x = np.array([[params[name] for name in config.SHAPE_PARAM_NAMES]], dtype=np.float64)
    return float(model.predict(x)[0])


# ------------------------------------------------------------------ #
# FRONTAL-AREA MODEL (second, small model)
# ------------------------------------------------------------------ #
# Air drag force is 0.5 * rho * (Cd * A) * v^2, so the energy model needs the
# drag AREA Cd * A, not Cd alone. Each design has its own frontal area A
# (Member 2's run table, column A_frontal). A is plain geometry, it moves
# smoothly with the shape values and has no pressure physics in it, so a small
# linear or quadratic ridge fit on the real designs is enough. No synthetic
# rows are used: the POD labeller can label Cd but not geometry.
AREA_CANDIDATE_BUILDERS = {
    "linear_ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-6, 2, 30))),
    "quadratic_ridge": lambda: make_pipeline(
        StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(),
        RidgeCV(alphas=np.logspace(-6, 2, 30)),
    ),
}


def train_area_model(train_df, holdout_df):
    """Fit the frontal-area candidates on the real training designs, score on
    the held-out real designs (mean and worst error in m2 and in percent), keep
    the best by hold-out mean error. Returns (best_name, best_model, results_df)."""
    x_train = train_df[config.SHAPE_PARAM_NAMES].to_numpy(dtype=np.float64)
    y_train = train_df["frontal_area"].to_numpy(dtype=np.float64)
    x_hold = holdout_df[config.SHAPE_PARAM_NAMES].to_numpy(dtype=np.float64)
    y_hold = holdout_df["frontal_area"].to_numpy(dtype=np.float64)
    rows, fitted = [], {}
    for name, build in AREA_CANDIDATE_BUILDERS.items():
        model = build().fit(x_train, y_train)
        fitted[name] = model
        err = np.abs(model.predict(x_hold) - y_hold)
        rows.append({"model": name, "holdout_mae_m2": float(err.mean()), "holdout_max_error_m2": float(err.max()),
                     "holdout_mae_percent": float(100 * (err / y_hold).mean()),
                     "holdout_max_error_percent": float(100 * (err / y_hold).max())})
    results = pd.DataFrame(rows).sort_values("holdout_mae_m2").reset_index(drop=True)
    best = results.iloc[0]["model"]
    return best, fitted[best], results


def save_area_model(model, name="frontal_area", out_dir=config.MODELS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_model.joblib")
    joblib.dump(model, path)
    return path


def predict_frontal_area(area_model, params):
    """Shape values dict -> predicted frontal area in m2."""
    x = np.array([[params[name] for name in config.SHAPE_PARAM_NAMES]], dtype=np.float64)
    return float(area_model.predict(x)[0])
