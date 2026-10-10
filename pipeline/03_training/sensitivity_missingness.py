"""
sensitivity_missingness.py
──────────────────────────
Missingness Sensitivity Analysis for Sparse Pathfinder Elements.

Evaluates model performance with versus without the four till geochemistry
elements that exhibit >50% null values at point sampling locations:
  • Bismuth  (Bi,  60.3% null)
  • Indium   (In,  60.3% null)
  • Thallium (Tl,  60.3% null)
  • Manganese(Mn,  58.3% null)

In BOTH configurations the spatially continuous IDW-interpolated surfaces
(geochem_bi_ppm_idw, geochem_in_ppm_idw, geochem_tl_ppm_idw,
geochem_mn_ppm_idw) are RETAINED.  Only the four raw point-geochemistry
columns (bi_ppm, in_ppm, tl_ppm, mn_ppm) are toggled.

Classifiers evaluated:
  • Random Forest  (published tuned hyperparameters from rf_cv_metrics.csv)
  • XGBoost        (published tuned hyperparameters from xgb_cv_metrics.csv)

Cross-validation: 5-fold spatial block CV, SMOTE within each training fold.

Outputs:
  models/sensitivity_missingness.csv  — Table 7 / Supplementary Table S2

Usage:
    python pipeline/03_training/sensitivity_missingness.py
"""

import sys
import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    auc as sklearn_auc,
)
from xgboost import XGBClassifier

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    FEATURE_MATRIX_PQ, PROCESSED_DIR, MODELS_DIR,
    ALL_FEATURES, RANDOM_STATE, N_SPATIAL_FOLDS,
    CLASS_WEIGHT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Sparse raw point columns to toggle ───────────────────────────────────────
SPARSE_RAW_COLS = ["bi_ppm", "in_ppm", "tl_ppm", "mn_ppm"]

MAX_NULL_FRACTION = 0.75   # feature-level null filter (same as other scripts)


# ── Load best published hyperparameters ──────────────────────────────────────

def _load_rf_best_params() -> dict:
    path = MODELS_DIR / "rf_cv_metrics.csv"
    if path.exists():
        row = pd.read_csv(path).iloc[0]
        params = ast.literal_eval(row["best_params"])
        params.update({"class_weight": CLASS_WEIGHT, "n_jobs": -1,
                        "random_state": RANDOM_STATE})
        log.info(f"RF best params loaded: {params}")
        return params
    raise FileNotFoundError(
        "rf_cv_metrics.csv not found — run train_rf.py first."
    )


def _load_xgb_best_params() -> dict:
    path = MODELS_DIR / "xgb_cv_metrics.csv"
    if path.exists():
        row = pd.read_csv(path).iloc[0]
        params = ast.literal_eval(row["best_params"])
        params.update({"random_state": RANDOM_STATE,
                       "eval_metric": "logloss", "verbosity": 0,
                       "n_jobs": -1})
        log.info(f"XGB best params loaded: {params}")
        return params
    raise FileNotFoundError(
        "xgb_cv_metrics.csv not found — run train_xgb.py first."
    )


# ── Spatial CV splitter ───────────────────────────────────────────────────────

class SpatialBlockCV:
    def __init__(self, fold_ids, n_splits=N_SPATIAL_FOLDS):
        self.fold_ids = fold_ids
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_total = len(X)
        n_orig  = len(self.fold_ids)
        for fold in range(self.n_splits):
            test_idx  = np.where(self.fold_ids == fold)[0]
            train_idx = np.where(self.fold_ids != fold)[0]
            if n_total > n_orig:
                train_idx = np.concatenate(
                    [train_idx, np.arange(n_orig, n_total)]
                )
            yield train_idx, test_idx


def _smote(X, y, seed):
    try:
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=seed).fit_resample(X, y)
    except ImportError:
        log.warning("imbalanced-learn not found — SMOTE skipped.")
        return X, y


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(clf, X, y, fold_ids):
    cv = SpatialBlockCV(fold_ids)
    aucs, aps, baccs, sr_aucs = [], [], [], []

    for fold, (tr, te) in enumerate(cv.split(X, y)):
        Xtr, ytr = _smote(X[tr], y[tr], RANDOM_STATE + fold)
        clf.fit(Xtr, ytr)
        yp    = clf.predict_proba(X[te])[:, 1]
        ypred = (yp >= 0.5).astype(int)

        aucs.append(roc_auc_score(y[te], yp))
        aps.append(average_precision_score(y[te], yp))
        baccs.append(balanced_accuracy_score(y[te], ypred))

        order    = np.argsort(yp)[::-1]
        sorted_y = y[te][order]
        n_pos    = sorted_y.sum()
        if n_pos > 0:
            cum_dep     = np.cumsum(sorted_y) / n_pos
            cum_samples = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
            sr_aucs.append(sklearn_auc(cum_samples, cum_dep))
        else:
            sr_aucs.append(0.5)

    return {
        "roc_auc_mean":      np.mean(aucs),    "roc_auc_std":      np.std(aucs),
        "avg_prec_mean":     np.mean(aps),     "avg_prec_std":     np.std(aps),
        "balanced_acc_mean": np.mean(baccs),   "balanced_acc_std": np.std(baccs),
        "sr_auc_mean":       np.mean(sr_aucs), "sr_auc_std":       np.std(sr_aucs),
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    fm = pd.read_parquet(FEATURE_MATRIX_PQ)
    labeled = fm[fm["label"].notna()].copy()
    labeled["label"] = labeled["label"].astype(int)

    spatial_folds = np.load(
        PROCESSED_DIR / "training_dataset" / "spatial_folds.npy"
    )
    assert len(spatial_folds) == len(labeled), (
        f"spatial_folds length mismatch: {len(spatial_folds)} vs {len(labeled)}"
    )

    pos_count = (labeled["label"] == 1).sum()
    neg_count = (labeled["label"] == 0).sum()
    log.info(f"Loaded {len(labeled)} labeled samples  "
             f"(pos={pos_count}, neg={neg_count})")
    return labeled, spatial_folds


def _build_Xy(labeled, spatial_folds, feat_cols):
    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(labeled[feat_cols].values.astype(np.float32))
    y   = labeled["label"].values.astype(int)
    fold_ids = spatial_folds[: len(labeled)]   # aligned by assertion above
    return X, y, fold_ids


# ── Missingness configurations ────────────────────────────────────────────────

def build_feature_sets(labeled):
    """Return [(label, feat_cols)] for 'with' and 'without' sparse raw cols."""
    available = [c for c in ALL_FEATURES if c in labeled.columns]
    null_f    = labeled[available].isnull().mean()
    base_cols = null_f[null_f <= MAX_NULL_FRACTION].index.tolist()

    # Configuration 1: with sparse raw columns (published)
    with_cols = base_cols

    # Configuration 2: without sparse raw point columns (IDW surfaces kept)
    without_cols = [c for c in base_cols if c not in SPARSE_RAW_COLS]

    sparse_present = [c for c in SPARSE_RAW_COLS if c in base_cols]
    log.info(f"Sparse raw cols present in feature set: {sparse_present}")
    log.info(f"Features WITH sparse cols   : {len(with_cols)}")
    log.info(f"Features WITHOUT sparse cols: {len(without_cols)}")

    return [
        ("With Bi/In/Tl/Mn raw columns (Published)",  with_cols),
        ("Without Bi/In/Tl/Mn raw columns",           without_cols),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 66)
    log.info(" Missingness Sensitivity Analysis - Sparse Pathfinder Elements")
    log.info("=" * 66)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rf_p  = _load_rf_best_params()
    xgb_p = _load_xgb_best_params()

    labeled, spatial_folds = load_data()
    configs = build_feature_sets(labeled)

    results = []

    for config_label, feat_cols in configs:
        log.info(f"\n{'-' * 60}")
        log.info(f"  Configuration: {config_label}")
        log.info(f"  Features: {len(feat_cols)}")

        X, y, fold_ids = _build_Xy(labeled, spatial_folds, feat_cols)
        log.info(f"  X shape: {X.shape}  pos={np.sum(y==1)}  neg={np.sum(y==0)}")

        # ── Random Forest ─────────────────────────────────────────────────
        log.info("  Running Random Forest ...")
        rf_clf = RandomForestClassifier(**rf_p)
        rf_m   = _evaluate(rf_clf, X, y, fold_ids)
        log.info(
            f"    RF  -> AUC={rf_m['roc_auc_mean']:.4f}+/-{rf_m['roc_auc_std']:.4f} | "
            f"AP={rf_m['avg_prec_mean']:.4f}+/-{rf_m['avg_prec_std']:.4f} | "
            f"BA={rf_m['balanced_acc_mean']:.4f}+/-{rf_m['balanced_acc_std']:.4f}"
        )

        # ── XGBoost ──────────────────────────────────────────────────────
        log.info("  Running XGBoost ...")
        xgb_clf = XGBClassifier(**xgb_p)
        xgb_m   = _evaluate(xgb_clf, X, y, fold_ids)
        log.info(
            f"    XGB -> AUC={xgb_m['roc_auc_mean']:.4f}+/-{xgb_m['roc_auc_std']:.4f} | "
            f"AP={xgb_m['avg_prec_mean']:.4f}+/-{xgb_m['avg_prec_std']:.4f} | "
            f"BA={xgb_m['balanced_acc_mean']:.4f}+/-{xgb_m['balanced_acc_std']:.4f}"
        )

        results.append({
            "config":                config_label,
            "n_features":            len(feat_cols),
            "sparse_cols_excluded":  config_label.startswith("Without"),
            **{f"rf_{k}":  v for k, v in rf_m.items()},
            **{f"xgb_{k}": v for k, v in xgb_m.items()},
        })

    df = pd.DataFrame(results)
    out_path = MODELS_DIR / "sensitivity_missingness.csv"
    df.to_csv(out_path, index=False)
    log.info(f"\nResults saved -> {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────
    log.info("\n====== SUMMARY: MISSINGNESS SENSITIVITY (RF & XGBoost) ======")
    log.info(
        f"{'Configuration':<45} | {'RF ROC-AUC':>16} | "
        f"{'RF Avg Prec':>14} | {'XGB ROC-AUC':>16} | {'XGB Avg Prec':>14}"
    )
    log.info("-" * 115)
    for _, r in df.iterrows():
        log.info(
            f"{r['config']:<45} | "
            f"{r['rf_roc_auc_mean']:.4f}+/-{r['rf_roc_auc_std']:.4f} | "
            f"{r['rf_avg_prec_mean']:.4f}+/-{r['rf_avg_prec_std']:.4f} | "
            f"{r['xgb_roc_auc_mean']:.4f}+/-{r['xgb_roc_auc_std']:.4f} | "
            f"{r['xgb_avg_prec_mean']:.4f}+/-{r['xgb_avg_prec_std']:.4f}"
        )


if __name__ == "__main__":
    main()
