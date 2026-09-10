"""Shared helpers for Phase 3/4: data loading, metrics, model factories."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

try:
    from training.features import FEATURE_COLUMNS
except ImportError:
    from features import FEATURE_COLUMNS

ABLATION_GROUPS = {
    "skills": ["skill_overlap", "n_res_skills", "n_jd_skills", "matched",
               "missing", "missing_ratio", "skill_idf_overlap"],
    "semantic": ["tfidf_cosine", "sbert_cosine", "emb_diff_mean", "emb_diff_std"],
    "experience": ["exp_score_norm", "exp_gap_norm", "exp_meets"],
    "education": ["edu_match", "edu_diff"],
    # length/track features stay in every ablation as neutral controls
}

REGISTRY = os.path.join(BACKEND, "models", "registry", "ensemble_v1")


def load_features(version="2"):
    import pandas as pd
    proc = os.path.join(BACKEND, "data", "processed")
    df = pd.read_parquet(os.path.join(proc, f"features_v{version}.parquet"))
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "val"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    return df, tr, va, te


def ece_score(y_true, y_prob, n_bins=10):
    import numpy as np
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if m.sum():
            ece += m.mean() * abs(y_prob[m].mean() - y_true[m].mean())
    return round(float(ece), 4)


def precision_at_k(df, score_col, k=5, min_group=10):
    """Mean P@k ranking resumes per JD (groups with >= min_group pairs)."""
    precisions = []
    for _, g in df.groupby("jd_id"):
        if len(g) < min_group:
            continue
        top = g.nlargest(k, score_col)
        precisions.append(top["label_match"].mean())
    return round(float(sum(precisions) / len(precisions)) if precisions else 0.0, 4), len(precisions)


def binary_metrics(y_true, y_prob, y_pred):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, roc_auc_score)
    return {
        "acc": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "ece": ece_score(y_true, y_prob),
    }


def make_m1():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced",
                                            max_iter=2000, random_state=42))


def make_m2():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                          max_leaf_nodes=31, min_samples_leaf=20,
                                          l2_regularization=1.0,
                                          class_weight="balanced", random_state=42)


def make_m3():
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                         learning_rate_init=1e-3, max_iter=60,
                         early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=10, random_state=42)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def pair_embedding_matrix(version="2", split_df=None):
    """Build [R_emb, J_emb, |R-J|] rows (1152-d) for pairs via unique-text lookup."""
    import numpy as np
    import pandas as pd
    proc = os.path.join(BACKEND, "data", "processed")
    z = np.load(os.path.join(proc, f"embeddings_v{version}.npz"), allow_pickle=True)
    r_map = {t: e for t, e in zip(z["resume_texts"].tolist(), z["resume_emb"])}
    j_map = {t: e for t, e in zip(z["jd_texts"].tolist(), z["jd_emb"])}
    pairs = pd.read_parquet(os.path.join(proc, f"pairs_v{version}.parquet"),
                            columns=["pair_id", "resume_text", "jd_text", "split"])
    if split_df is not None:
        pairs = pairs[pairs["pair_id"].isin(set(split_df["pair_id"]))].reset_index(drop=True)
        pairs = pairs.set_index("pair_id").loc[split_df["pair_id"]].reset_index()
    R = np.stack([r_map[t] for t in pairs["resume_text"]]).astype("float32")
    J = np.stack([j_map[t] for t in pairs["jd_text"].astype(str)]).astype("float32")
    return np.concatenate([R, J, np.abs(R - J)], axis=1).astype("float32")
