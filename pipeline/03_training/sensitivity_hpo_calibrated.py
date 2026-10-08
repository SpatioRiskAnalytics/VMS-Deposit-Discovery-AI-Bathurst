"""
sensitivity_hpo_calibrated.py
─────────────────────────────
Full Optuna Hyperparameter Optimization (50 trials) and Fold-Level
Threshold Calibration for the Negative Label Sensitivity Analysis (R1-C1, R2-C8).

Evaluates across 5-fold spatial block cross-validation with:
  • Optuna Bayesian HPO (50 trials per config) for both RF and XGBoost
  • SMOTE applied strictly inside each training fold (no leakage)
  • Fold-level decision threshold calibration (maximizing training Balanced Accuracy)
  • Full metrics: ROC-AUC, Average Precision, Calibrated Balanced Accuracy, SR-AUC

Outputs:
  models/sensitivity_neg_ratio_hpo.csv  — Table S1 (Full HPO + Calibrated)

Usage:
  python pipeline/03_training/sensitivity_hpo_calibrated.py
"""

import sys
import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads as wkt_loads
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    auc as sklearn_auc
)
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    FEATURE_MATRIX_PQ, PROCESSED_DIR, MODELS_DIR,
    ALL_FEATURES, RANDOM_STATE, N_SPATIAL_FOLDS,
    BARREN_LABELS_GPKG, CLASS_WEIGHT, N_OPTUNA_TRIALS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MAX_NULL_FRACTION = 0.75
N_TRIALS = N_OPTUNA_TRIALS  # 50 trials per configuration

# ── Spatial CV Splitter ───────────────────────────────────────────────────────

class SpatialBlockCV:
    def __init__(self, fold_ids, n_splits=5):
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
                train_idx = np.concatenate([train_idx, np.arange(n_orig, n_total)])
            yield train_idx, test_idx


def _smote(X, y, seed):
    try:
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=seed).fit_resample(X, y)
    except ImportError:
        return X, y


def _find_optimal_threshold(y_true, y_prob):
    """Find decision threshold in [0.05, 0.95] maximizing Balanced Accuracy on training set."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_ba = -1.0
    best_t = 0.5
    for t in thresholds:
        ba = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if ba > best_ba:
            best_ba = ba
            best_t = t
    return best_t


# ── Data Loading & Preparation ────────────────────────────────────────────────

def load_base_data():
    fm   = pd.read_parquet(FEATURE_MATRIX_PQ)
    gpkg = gpd.read_file(BARREN_LABELS_GPKG)

    labeled = fm[fm["label"].notna()].copy()
    labeled["label"] = labeled["label"].astype(int)

    def _round_key(wkt_str, decimals=0):
        pt = wkt_loads(wkt_str)
        return (round(pt.x, decimals), round(pt.y, decimals))

    gpkg["_geo_key"] = gpkg.geometry.apply(
        lambda g: (round(g.x, 0), round(g.y, 0))
    )
    geo_src_map = gpkg.set_index("_geo_key")["source"].to_dict()

    neg_mask = labeled["label"] == 0
    labeled.loc[neg_mask, "_geo_key"] = labeled.loc[
        neg_mask, "geometry_wkt"
    ].apply(_round_key)
    labeled.loc[neg_mask, "source_detail"] = labeled.loc[
        neg_mask, "_geo_key"
    ].map(geo_src_map)

    fm_pos    = labeled[labeled["label"] == 1]
    fm_barren = labeled[labeled["source_detail"] == "barren_hole_geonb"]
    fm_mah    = labeled[labeled["source_detail"] == "pseudo_absence_dissimilar"]

    log.info(f"Loaded: pos={len(fm_pos)}  barren={len(fm_barren)}  mah={len(fm_mah)}")

    spatial_folds = np.load(PROCESSED_DIR / "training_dataset" / "spatial_folds.npy")
    assert len(spatial_folds) == len(labeled)

    return fm_pos, fm_barren, fm_mah, labeled, spatial_folds


def _feat_cols(labeled):
    available = [c for c in ALL_FEATURES if c in labeled.columns]
    null_f    = labeled[available].isnull().mean()
    return null_f[null_f <= MAX_NULL_FRACTION].index.tolist()


def _build_Xy(fm_pos, neg_df, labeled_full, spatial_folds, feat_cols):
    subset = pd.concat([fm_pos, neg_df], ignore_index=False)
    imp    = SimpleImputer(strategy="median")
    X = imp.fit_transform(subset[feat_cols].values.astype(np.float32))
    y = subset["label"].values.astype(int)

    rank_map = {orig: rank for rank, orig in enumerate(labeled_full.index)}
    fold_ids = np.array([spatial_folds[rank_map[i]] for i in subset.index])
    return X, y, fold_ids


# ── Optuna HPO ────────────────────────────────────────────────────────────────

def optimize_rf(X, y, fold_ids, n_trials=N_TRIALS):
    cv = SpatialBlockCV(fold_ids, N_SPATIAL_FOLDS)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            "class_weight":     CLASS_WEIGHT,
            "n_jobs":           -1,
            "random_state":     RANDOM_STATE,
        }
        clf = RandomForestClassifier(**params)
        scores = []
        for tr, te in cv.split(X, y):
            Xtr, ytr = _smote(X[tr], y[tr], RANDOM_STATE)
            clf.fit(Xtr, ytr)
            yp = clf.predict_proba(X[te])[:, 1]
            scores.append(roc_auc_score(y[te], yp))
        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials)
    best_p = study.best_params
    best_p.update({
        "class_weight": CLASS_WEIGHT,
        "n_jobs": -1,
        "random_state": RANDOM_STATE
    })
    return best_p, study.best_value


def optimize_xgb(X, y, fold_ids, n_trials=N_TRIALS):
    cv = SpatialBlockCV(fold_ids, N_SPATIAL_FOLDS)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "random_state":     RANDOM_STATE,
            "eval_metric":      "logloss",
            "verbosity":        0,
            "n_jobs":           -1,
        }
        clf = XGBClassifier(**params)
        scores = []
        for tr, te in cv.split(X, y):
            Xtr, ytr = _smote(X[tr], y[tr], RANDOM_STATE)
            clf.fit(Xtr, ytr)
            yp = clf.predict_proba(X[te])[:, 1]
            scores.append(roc_auc_score(y[te], yp))
        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials)
    best_p = study.best_params
    best_p.update({
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "verbosity": 0,
        "n_jobs": -1,
    })
    return best_p, study.best_value


# ── Calibrated Evaluation ─────────────────────────────────────────────────────

def evaluate_calibrated(clf_class, best_params, X, y, fold_ids):
    cv = SpatialBlockCV(fold_ids, N_SPATIAL_FOLDS)
    aucs, aps, baccs_cal, baccs_def, sr_aucs = [], [], [], [], []

    for fold, (tr, te) in enumerate(cv.split(X, y)):
        Xtr, ytr = _smote(X[tr], y[tr], RANDOM_STATE + fold)
        clf = clf_class(**best_params)
        clf.fit(Xtr, ytr)

        # Predict train probabilities to calibrate threshold
        yp_tr = clf.predict_proba(Xtr)[:, 1]
        t_opt = _find_optimal_threshold(ytr, yp_tr)

        # Predict test probabilities
        yp_te = clf.predict_proba(X[te])[:, 1]
        y_true = y[te]

        # Calculate metrics
        aucs.append(roc_auc_score(y_true, yp_te))
        aps.append(average_precision_score(y_true, yp_te))
        baccs_def.append(balanced_accuracy_score(y_true, (yp_te >= 0.5).astype(int)))
        baccs_cal.append(balanced_accuracy_score(y_true, (yp_te >= t_opt).astype(int)))

        # Success-Rate AUC (cumulative deposit capture vs cumulative samples)
        order = np.argsort(yp_te)[::-1]
        sorted_y = y_true[order]
        n_pos = sorted_y.sum()
        if n_pos > 0:
            cum_dep = np.cumsum(sorted_y) / n_pos
            cum_samples = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
            sr_aucs.append(sklearn_auc(cum_samples, cum_dep))
        else:
            sr_aucs.append(0.5)

    return {
        "roc_auc_mean": np.mean(aucs),          "roc_auc_std": np.std(aucs),
        "avg_prec_mean": np.mean(aps),          "avg_prec_std": np.std(aps),
        "bal_acc_cal_mean": np.mean(baccs_cal), "bal_acc_cal_std": np.std(baccs_cal),
        "bal_acc_def_mean": np.mean(baccs_def), "bal_acc_def_std": np.std(baccs_def),
        "sr_auc_mean": np.mean(sr_aucs),        "sr_auc_std": np.std(sr_aucs),
    }


# ── Configurations ────────────────────────────────────────────────────────────

RATIO_CONFIGS = [
    ("A: Pure random (control)",               0,   0,   True),
    ("B: 0 barren / 250 Mahalanobis",          0,   250, False),
    ("C: 63 barren / 187 Mahalanobis",         63,  187, False),
    ("D: 125 barren / 125 Mahal. (published)", 125, 125, False),
    ("E: 187 barren / 63 Mahalanobis",         187, 63,  False),
    ("F: 250 barren / 0 Mahalanobis",          250, 0,   False),
]


def main():
    log.info("══════════════════════════════════════════════════════════════")
    log.info(" Full Optuna HPO (50 trials) & Calibrated Sensitivity Analysis")
    log.info("══════════════════════════════════════════════════════════════")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    fm_pos, fm_barren, fm_mah, labeled_full, spatial_folds = load_base_data()
    feat_cols = _feat_cols(labeled_full)
    log.info(f"Using {len(feat_cols)} features for evaluation.")

    results = []

    for idx, (lbl, n_barren, n_mah, is_rand) in enumerate(RATIO_CONFIGS, 1):
        log.info(f"\n[{idx}/6] Configuration: {lbl}")

        if is_rand:
            all_neg = pd.concat([fm_barren, fm_mah])
            chosen  = all_neg.sample(n=min(250, len(all_neg)),
                                     random_state=RANDOM_STATE, replace=False)
        else:
            parts = []
            if n_barren > 0:
                parts.append(fm_barren.sample(n=min(n_barren, len(fm_barren)),
                                              random_state=RANDOM_STATE))
            if n_mah > 0:
                parts.append(fm_mah.sample(n=min(n_mah, len(fm_mah)),
                                           random_state=RANDOM_STATE))
            chosen = pd.concat(parts)

        X, y, fold_ids = _build_Xy(fm_pos, chosen, labeled_full,
                                   spatial_folds, feat_cols)
        log.info(f"  Dataset: N={len(y)} (pos={np.sum(y==1)}, neg={np.sum(y==0)})")

        # 1. Random Forest Optuna HPO & Calibrated Evaluation
        log.info(f"  Running Optuna RF ({N_TRIALS} trials)...")
        rf_best_p, rf_val = optimize_rf(X, y, fold_ids, n_trials=N_TRIALS)
        log.info(f"    RF Best CV ROC-AUC: {rf_val:.4f} with {rf_best_p}")
        rf_metrics = evaluate_calibrated(RandomForestClassifier, rf_best_p, X, y, fold_ids)

        # 2. XGBoost Optuna HPO & Calibrated Evaluation
        log.info(f"  Running Optuna XGBoost ({N_TRIALS} trials)...")
        xgb_best_p, xgb_val = optimize_xgb(X, y, fold_ids, n_trials=N_TRIALS)
        log.info(f"    XGB Best CV ROC-AUC: {xgb_val:.4f}")
        xgb_metrics = evaluate_calibrated(XGBClassifier, xgb_best_p, X, y, fold_ids)

        log.info(
            f"  -> RF : AUC={rf_metrics['roc_auc_mean']:.4f}±{rf_metrics['roc_auc_std']:.4f} | "
            f"BA(cal)={rf_metrics['bal_acc_cal_mean']:.4f} | "
            f"SR-AUC={rf_metrics['sr_auc_mean']:.4f} | "
            f"AP={rf_metrics['avg_prec_mean']:.4f}"
        )
        log.info(
            f"  -> XGB: AUC={xgb_metrics['roc_auc_mean']:.4f}±{xgb_metrics['roc_auc_std']:.4f} | "
            f"BA(cal)={xgb_metrics['bal_acc_cal_mean']:.4f} | "
            f"SR-AUC={xgb_metrics['sr_auc_mean']:.4f} | "
            f"AP={xgb_metrics['avg_prec_mean']:.4f}"
        )

        res_row = {
            "config": lbl,
            "n_barren": n_barren,
            "n_mah": n_mah,
            "is_random_control": is_rand,
            "rf_best_params": str(rf_best_p),
            "xgb_best_params": str(xgb_best_p),
            **{f"rf_{k}": v for k, v in rf_metrics.items()},
            **{f"xgb_{k}": v for k, v in xgb_metrics.items()}
        }
        results.append(res_row)

    df_res = pd.DataFrame(results)
    out_path = MODELS_DIR / "sensitivity_neg_ratio_hpo.csv"
    df_res.to_csv(out_path, index=False)
    log.info(f"\n✅ All 6 configurations complete! Table saved to: {out_path}")

    # Summary table
    log.info("\n══════════ SUMMARY TABLE (FULL 50-TRIAL HPO + CALIBRATED) ══════════")
    for _, r in df_res.iterrows():
        log.info(
            f"{r['config']:<42} | "
            f"RF: AUC={r['rf_roc_auc_mean']:.4f} BA={r['rf_bal_acc_cal_mean']:.4f} SR={r['rf_sr_auc_mean']:.4f} | "
            f"XGB: AUC={r['xgb_roc_auc_mean']:.4f} BA={r['xgb_bal_acc_cal_mean']:.4f} SR={r['xgb_sr_auc_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
