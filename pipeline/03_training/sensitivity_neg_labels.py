"""
sensitivity_neg_labels.py
─────────────────────────
Sensitivity analysis for the hybrid negative label strategy (R1-C1, R2-C8).

Experiment A — Ratio sensitivity (Table S1):
  Vary the barren-hole : Mahalanobis split across 6 configurations plus a
  pure random background control. Fixed best RF and XGB hyperparameters,
  SMOTE within fold, 5-fold spatial block CV.

Experiment B — Guard-distance sensitivity (Table S2):
  Hold the published 125/125 ratio fixed; vary the geographic guard distance
  (min_buffer_m) across: 500 m, 1,000 m, 2,000 m.

Outputs:
  models/sensitivity_neg_ratio.csv    — Table S1 (ratio sweep)
  models/sensitivity_guard_dist.csv   — Table S2 (guard distance sweep)

Usage:
    python pipeline/03_training/sensitivity_neg_labels.py
"""

import sys
import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
)
from xgboost import XGBClassifier

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))
from config import (
    FEATURE_MATRIX_PQ, PROCESSED_DIR, MODELS_DIR,
    ALL_FEATURES, RANDOM_STATE, N_SPATIAL_FOLDS,
    BARREN_LABELS_GPKG, CLASS_WEIGHT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MAX_NULL_FRACTION = 0.75

# ── Load best hyperparameters from saved CSV ──────────────────────────────────

def _load_rf_best_params() -> dict:
    path = MODELS_DIR / "rf_cv_metrics.csv"
    if path.exists():
        row = pd.read_csv(path).iloc[0]
        params = ast.literal_eval(row["best_params"])
        params.update({"class_weight": CLASS_WEIGHT, "n_jobs": -1,
                       "random_state": RANDOM_STATE})
        log.info(f"RF best params loaded: {params}")
        return params
    raise FileNotFoundError("rf_cv_metrics.csv not found.")

def _load_xgb_best_params() -> dict:
    path = MODELS_DIR / "xgb_cv_metrics.csv"
    if path.exists():
        row = pd.read_csv(path).iloc[0]
        params = ast.literal_eval(row["best_params"])
        params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss",
                       "verbosity": 0})
        log.info(f"XGB best params loaded: {params}")
        return params
    raise FileNotFoundError("xgb_cv_metrics.csv not found.")

# ── Spatial CV splitter ───────────────────────────────────────────────────────

class SpatialBlockCV:
    def __init__(self, fold_ids, n_splits):
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

def _evaluate(clf, X, y, fold_ids, seed):
    cv = SpatialBlockCV(fold_ids, N_SPATIAL_FOLDS)
    aucs, aps, baccs = [], [], []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        Xtr, ytr = _smote(X[tr], y[tr], seed + fold)
        clf.fit(Xtr, ytr)
        yp   = clf.predict_proba(X[te])[:, 1]
        ypred = (yp >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], yp))
        aps.append(average_precision_score(y[te], yp))
        baccs.append(balanced_accuracy_score(y[te], ypred))
    return {
        "roc_auc_mean": np.mean(aucs),  "roc_auc_std": np.std(aucs),
        "avg_prec_mean": np.mean(aps),  "avg_prec_std": np.std(aps),
        "balanced_acc_mean": np.mean(baccs), "balanced_acc_std": np.std(baccs),
    }

# ── Data loading ──────────────────────────────────────────────────────────────

def load_base_data():
    fm   = pd.read_parquet(FEATURE_MATRIX_PQ)
    gpkg = gpd.read_file(BARREN_LABELS_GPKG)

    labeled = fm[fm["label"].notna()].copy()
    labeled["label"] = labeled["label"].astype(int)

    # The FM uses synthetic IDs (BARREN_0…N) that don't match GPKG hole_id.
    # Join by rounded geometry coordinates instead (confirmed to match).
    from shapely.wkt import loads as wkt_loads

    def _round_key(wkt_str, decimals=0):
        pt = wkt_loads(wkt_str)
        return (round(pt.x, decimals), round(pt.y, decimals))

    gpkg["_geo_key"] = gpkg.geometry.apply(
        lambda g: (round(g.x, 0), round(g.y, 0))
    )
    geo_src_map   = gpkg.set_index("_geo_key")["source"].to_dict()
    geo_mahal_map = gpkg.set_index("_geo_key")["mahal_dist"].to_dict()

    neg_mask = labeled["label"] == 0
    labeled.loc[neg_mask, "_geo_key"] = labeled.loc[
        neg_mask, "geometry_wkt"
    ].apply(_round_key)

    labeled.loc[neg_mask, "source_detail"] = labeled.loc[
        neg_mask, "_geo_key"
    ].map(geo_src_map)
    labeled.loc[neg_mask, "mahal_dist"] = labeled.loc[
        neg_mask, "_geo_key"
    ].map(geo_mahal_map)

    n_matched = labeled.loc[neg_mask, "source_detail"].notna().sum()
    log.info(f"Geometry-join matched {n_matched}/{neg_mask.sum()} negatives to GPKG source")

    fm_pos    = labeled[labeled["label"] == 1]
    fm_barren = labeled[labeled["source_detail"] == "barren_hole_geonb"]
    fm_mah    = labeled[labeled["source_detail"] == "pseudo_absence_dissimilar"]

    log.info(f"pos={len(fm_pos)}  barren={len(fm_barren)}  mah={len(fm_mah)}")

    spatial_folds = np.load(PROCESSED_DIR / "training_dataset" / "spatial_folds.npy")
    assert len(spatial_folds) == len(labeled)

    return fm_pos, fm_barren, fm_mah, labeled, spatial_folds
    fm   = pd.read_parquet(FEATURE_MATRIX_PQ)
    gpkg = gpd.read_file(BARREN_LABELS_GPKG)

    labeled = fm[fm["label"].notna()].copy()
    labeled["label"] = labeled["label"].astype(int)

    # Attach source_detail and mahal_dist from GPKG via hole_id -> point_id
    src_map   = gpkg.set_index("hole_id")["source"].to_dict()
    mahal_map = gpkg.set_index("hole_id")["mahal_dist"].to_dict()
    neg_mask  = labeled["label"] == 0
    labeled.loc[neg_mask, "source_detail"] = labeled.loc[neg_mask, "point_id"].map(src_map)
    labeled.loc[neg_mask, "mahal_dist"]    = labeled.loc[neg_mask, "point_id"].map(mahal_map)

    fm_pos    = labeled[labeled["label"] == 1]
    fm_barren = labeled[labeled["source_detail"] == "barren_hole_geonb"]
    fm_mah    = labeled[labeled["source_detail"] == "pseudo_absence_dissimilar"]

    log.info(f"pos={len(fm_pos)}  barren={len(fm_barren)}  mah={len(fm_mah)}")

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

# ── Experiment A ──────────────────────────────────────────────────────────────

RATIO_CONFIGS = [
    ("A: Pure random (control)",              0,   0,   True),
    ("B: 0 barren / 250 Mahalanobis",         0,   250, False),
    ("C: 63 barren / 187 Mahalanobis",        63,  187, False),
    ("D: 125 barren / 125 Mahal. (published)",125, 125, False),
    ("E: 187 barren / 63 Mahalanobis",        187, 63,  False),
    ("F: 250 barren / 0 Mahalanobis",         250, 0,   False),
]

def run_ratio_sensitivity(fm_pos, fm_barren, fm_mah, labeled_full,
                          spatial_folds, rf_p, xgb_p):
    log.info("\n══ Experiment A: Ratio sensitivity ══")
    feat_cols = _feat_cols(labeled_full)
    results = []

    for lbl, n_barren, n_mah, is_rand in RATIO_CONFIGS:
        log.info(f"\n── {lbl} ──")
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
        log.info(f"  X={X.shape} y={dict(zip(*np.unique(y, return_counts=True)))}")

        rf_m  = _evaluate(RandomForestClassifier(**rf_p), X, y, fold_ids, RANDOM_STATE)
        xgb_m = _evaluate(XGBClassifier(**xgb_p), X, y, fold_ids, RANDOM_STATE)

        log.info(f"  RF  AUC={rf_m['roc_auc_mean']:.4f}±{rf_m['roc_auc_std']:.4f}"
                 f"  AP={rf_m['avg_prec_mean']:.4f}")
        log.info(f"  XGB AUC={xgb_m['roc_auc_mean']:.4f}±{xgb_m['roc_auc_std']:.4f}"
                 f"  AP={xgb_m['avg_prec_mean']:.4f}")

        results.append({"config": lbl, "n_barren": n_barren, "n_mah": n_mah,
                        "is_random_control": is_rand,
                        **{f"rf_{k}": v for k, v in rf_m.items()},
                        **{f"xgb_{k}": v for k, v in xgb_m.items()}})

    return pd.DataFrame(results)

# ── Experiment B ──────────────────────────────────────────────────────────────

GUARD_DISTANCES_M = [500, 1000, 2000]

def run_guard_sensitivity(fm_pos, fm_barren, fm_mah, labeled_full,
                          spatial_folds, rf_p, xgb_p):
    log.info("\n══ Experiment B: Guard-distance sensitivity (125/125) ══")
    feat_cols = _feat_cols(labeled_full)
    results   = []

    gpkg_neg = gpd.read_file(BARREN_LABELS_GPKG)
    gpkg_pos = gpd.read_file(Path("data/raw/labels/vms_positive_labels.gpkg"))
    mah_gdf  = gpkg_neg[gpkg_neg["source"] == "pseudo_absence_dissimilar"].copy()
    dep_union = gpkg_pos.geometry.unary_union
    mah_gdf["min_dist_m"] = mah_gdf.geometry.distance(dep_union)

    for guard_m in GUARD_DISTANCES_M:
        lbl = f"Guard = {guard_m} m"
        log.info(f"\n── {lbl} ──")
        eligible_ids  = mah_gdf.loc[mah_gdf["min_dist_m"] >= guard_m, "hole_id"].tolist()
        fm_mah_elig   = fm_mah[fm_mah["point_id"].isin(eligible_ids)]
        n_mah = min(125, len(fm_mah_elig))
        n_bar = min(125, len(fm_barren))
        log.info(f"  Eligible MAH={len(fm_mah_elig)} → using {n_mah}; barren={n_bar}")

        chosen = pd.concat([
            fm_barren.sample(n=n_bar, random_state=RANDOM_STATE),
            fm_mah_elig.sample(n=n_mah, random_state=RANDOM_STATE),
        ])

        X, y, fold_ids = _build_Xy(fm_pos, chosen, labeled_full,
                                   spatial_folds, feat_cols)

        rf_m  = _evaluate(RandomForestClassifier(**rf_p), X, y, fold_ids, RANDOM_STATE)
        xgb_m = _evaluate(XGBClassifier(**xgb_p), X, y, fold_ids, RANDOM_STATE)

        log.info(f"  RF  AUC={rf_m['roc_auc_mean']:.4f}±{rf_m['roc_auc_std']:.4f}"
                 f"  AP={rf_m['avg_prec_mean']:.4f}")
        log.info(f"  XGB AUC={xgb_m['roc_auc_mean']:.4f}±{xgb_m['roc_auc_std']:.4f}"
                 f"  AP={xgb_m['avg_prec_mean']:.4f}")

        results.append({"config": lbl, "guard_dist_m": guard_m,
                        "n_barren": n_bar, "n_mah": n_mah,
                        **{f"rf_{k}": v for k, v in rf_m.items()},
                        **{f"xgb_{k}": v for k, v in xgb_m.items()}})

    return pd.DataFrame(results)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("Negative Label Sensitivity Analysis — R1-C1 / R2-C8")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rf_p  = _load_rf_best_params()
    xgb_p = _load_xgb_best_params()
    fm_pos, fm_barren, fm_mah, labeled_full, spatial_folds = load_base_data()

    df_ratio = run_ratio_sensitivity(
        fm_pos, fm_barren, fm_mah, labeled_full, spatial_folds, rf_p, xgb_p)
    out_a = MODELS_DIR / "sensitivity_neg_ratio.csv"
    df_ratio.to_csv(out_a, index=False)
    log.info(f"\n✅ Table S1 saved → {out_a}")

    df_guard = run_guard_sensitivity(
        fm_pos, fm_barren, fm_mah, labeled_full, spatial_folds, rf_p, xgb_p)
    out_b = MODELS_DIR / "sensitivity_guard_dist.csv"
    df_guard.to_csv(out_b, index=False)
    log.info(f"✅ Table S2 saved → {out_b}")

    # Summary
    log.info("\n═══ TABLE S1 — Ratio sensitivity ═══")
    for _, r in df_ratio.iterrows():
        log.info(f"  {r['config']:<44} | RF AUC={r['rf_roc_auc_mean']:.4f}±{r['rf_roc_auc_std']:.4f}"
                 f" AP={r['rf_avg_prec_mean']:.4f} | XGB AUC={r['xgb_roc_auc_mean']:.4f}±{r['xgb_roc_auc_std']:.4f}"
                 f" AP={r['xgb_avg_prec_mean']:.4f}")

    log.info("\n═══ TABLE S2 — Guard-distance sensitivity ═══")
    for _, r in df_guard.iterrows():
        log.info(f"  {r['config']:<20} | RF AUC={r['rf_roc_auc_mean']:.4f}±{r['rf_roc_auc_std']:.4f}"
                 f" AP={r['rf_avg_prec_mean']:.4f} | XGB AUC={r['xgb_roc_auc_mean']:.4f}±{r['xgb_roc_auc_std']:.4f}"
                 f" AP={r['xgb_avg_prec_mean']:.4f}")

if __name__ == "__main__":
    main()
