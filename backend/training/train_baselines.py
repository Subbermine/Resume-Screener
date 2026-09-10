"""Phase 3: train base models M1/M2/M3 on features_v2 (full train, eval test).

M1: TF-IDF+structured compact features -> LogisticRegression (baseline)
M2: compact features -> HistGradientBoosting (structured GBM)
M3: SBERT embeddings [R, J, |R-J|] -> StandardScaler + MLP (semantic head)

Usage (from backend/): python -m training.train_baselines [--version 2]
Saves models + metrics_baselines.json into models/registry/ensemble_v1/.
Phase 4 reuses these as refit full-train Level-0 models (meta trains on OOF).
"""
import argparse
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

try:
    from training.features import FEATURE_COLUMNS
    from training.utils import (REGISTRY, binary_metrics, load_features, make_m1,
                                make_m2, make_m3, pair_embedding_matrix,
                                precision_at_k, save_json)
except ImportError:
    from features import FEATURE_COLUMNS
    from utils import (REGISTRY, binary_metrics, load_features, make_m1, make_m2,
                       make_m3, pair_embedding_matrix, precision_at_k, save_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="2", choices=["1", "2"])
    args = ap.parse_args()
    os.makedirs(REGISTRY, exist_ok=True)

    df, tr, va, te = load_features(args.version)
    Xtr, ytr = tr[FEATURE_COLUMNS].to_numpy(), tr["label_match"].to_numpy()
    Xte, yte = te[FEATURE_COLUMNS].to_numpy(), te["label_match"].to_numpy()

    out = {"version": args.version, "n_train": len(tr), "n_test": len(te), "models": {}}
    t0 = time.time()

    m1 = make_m1().fit(Xtr, ytr)
    p1 = m1.predict_proba(Xte)[:, 1]
    out["models"]["m1_logreg"] = {**binary_metrics(yte, p1, (p1 >= 0.5).astype(int)),
                                  **dict(zip(["p_at_5", "p_at_5_groups"],
                                             precision_at_k(te.assign(score=p1), "score")))}
    with open(os.path.join(REGISTRY, "m1_logreg.pkl"), "wb") as f:
        pickle.dump(m1, f)
    print("M1", out["models"]["m1_logreg"])

    m2 = make_m2().fit(Xtr, ytr)
    p2 = m2.predict_proba(Xte)[:, 1]
    out["models"]["m2_hgb"] = {**binary_metrics(yte, p2, (p2 >= 0.5).astype(int)),
                               **dict(zip(["p_at_5", "p_at_5_groups"],
                                          precision_at_k(te.assign(score=p2), "score")))}
    with open(os.path.join(REGISTRY, "hgb_m2.pkl"), "wb") as f:
        pickle.dump(m2, f)
    print("M2", out["models"]["m2_hgb"])

    from sklearn.preprocessing import StandardScaler
    Etr = pair_embedding_matrix(args.version, tr)
    Ete = pair_embedding_matrix(args.version, te)
    sc = StandardScaler().fit(Etr)
    m3 = make_m3().fit(sc.transform(Etr), ytr)
    p3 = m3.predict_proba(sc.transform(Ete))[:, 1]
    out["models"]["m3_mlp_emb"] = {**binary_metrics(yte, p3, (p3 >= 0.5).astype(int)),
                                   **dict(zip(["p_at_5", "p_at_5_groups"],
                                              precision_at_k(te.assign(score=p3), "score"))),
                                   "emb_dim": int(Etr.shape[1])}
    with open(os.path.join(REGISTRY, "mlp_m3.pkl"), "wb") as f:
        pickle.dump(m3, f)
    with open(os.path.join(REGISTRY, "scaler_m3.pkl"), "wb") as f:
        pickle.dump(sc, f)
    print("M3", out["models"]["m3_mlp_emb"])

    # Heuristic reference: legacy ats_proxy as score, label at >= 60.
    import numpy as np
    hprob = (te["ats_proxy"].to_numpy() / 100.0).clip(0, 1)
    out["models"]["heuristic_ats"] = {**binary_metrics(yte, hprob, (hprob >= 0.6).astype(int)),
                                      **dict(zip(["p_at_5", "p_at_5_groups"],
                                                 precision_at_k(te.assign(score=hprob), "score")))}
    print("heuristic", out["models"]["heuristic_ats"])

    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, os.path.join(REGISTRY, "metrics_baselines.json"))
    print(f"saved -> {REGISTRY} ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
