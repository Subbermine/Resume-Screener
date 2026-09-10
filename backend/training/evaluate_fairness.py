"""Phase 5 (fairness): slice test metrics by source / education / experience / track.

Usage (from backend/): python -m training.evaluate_fairness [--version 2]
Saves data/processed/fairness_v<V>.json. Flags slices with < 50 rows or with
selection-rate ratio < 0.8 vs the best slice (adverse-impact style check).
"""
import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def stack_proba(feat, version):
    import numpy as np
    import pandas as pd
    from training.features import FEATURE_COLUMNS
    from training.utils import REGISTRY, pair_embedding_matrix
    proc = os.path.join(BACKEND, "data", "processed")
    te = feat[feat["split"] == "test"].reset_index(drop=True)
    X = te[FEATURE_COLUMNS].to_numpy()
    E = pair_embedding_matrix(version, te)

    def load(n):
        with open(os.path.join(REGISTRY, n), "rb") as f:
            return pickle.load(f)
    m1, m2, m3, sc3, meta, cal = (load(n) for n in
        ["m1_logreg.pkl", "hgb_m2.pkl", "mlp_m3.pkl", "scaler_m3.pkl",
         "meta_logreg.pkl", "calibrator.pkl"])
    F = np.column_stack([m1.predict_proba(X)[:, 1], m2.predict_proba(X)[:, 1],
                         m3.predict_proba(sc3.transform(E))[:, 1],
                         te["sbert_cosine"].to_numpy() / 100.0,
                         te["skill_overlap"].to_numpy()])
    uncal = meta.predict_proba(F)[:, 1]
    return te, cal.predict_proba(uncal.reshape(-1, 1))[:, 1]


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="2", choices=["1", "2"])
    args = ap.parse_args()
    import pandas as pd
    from sklearn.metrics import f1_score, roc_auc_score
    feat = pd.read_parquet(os.path.join(BACKEND, "data", "processed", f"features_v{args.version}.parquet"))
    pairs = pd.read_parquet(os.path.join(BACKEND, "data", "processed", f"pairs_v{args.version}.parquet"),
                            columns=["pair_id", "edu_req", "exp_years_req", "track_jd"] +
                            (["kaggle_category"] if args.version == "2" else []))
    te, proba = stack_proba(feat, args.version)
    te = te.merge(pairs, on="pair_id", how="left").assign(score=proba)
    te["pred"] = (te["score"] >= 0.45).astype(int)
    te["exp_band"] = pd.cut(te["exp_years_req"], [-1, 1, 3, 100],
                            labels=["entry(<=1y)", "mid(2-3y)", "senior(4y+)"])

    slices = {"source": "source", "edu_req": "edu_req", "exp_band": "exp_band",
              "track_jd": "track_jd"} if args.version == "2" else \
             {"edu_req": "edu_req", "exp_band": "exp_band", "track_jd": "track_jd"}
    report = {}
    for name, col in slices.items():
        rows = {}
        for val, g in te.groupby(col, observed=True):
            if len(g) < 50:
                continue
            rows[str(val)] = {"n": int(len(g)),
                              "selection_rate": round(float(g["pred"].mean()), 4),
                              "f1": round(float(f1_score(g["label_match"], g["pred"])), 4),
                              "auc": round(float(roc_auc_score(g["label_match"], g["score"]))
                                           if g["label_match"].nunique() == 2 else 1.0, 4)}
        best = max((r["selection_rate"] for r in rows.values()), default=0)
        for v, r in rows.items():
            r["flag"] = bool(best and r["selection_rate"] / best < 0.8)
        report[name] = rows

    import json
    out = os.path.join(BACKEND, "data", "processed", f"fairness_v{args.version}.json")
    json.dump(report, open(out, "w"), indent=2)
    for name, rows in report.items():
        print(f"## {name}")
        for v, r in sorted(rows.items()):
            print(f"  {v:>16} n={r['n']:>5} sel={r['selection_rate']:.3f} f1={r['f1']:.3f} "
                  f"auc={r['auc']:.3f}" + ("  <-- FLAG" if r["flag"] else ""))
    print("saved ->", out)


if __name__ == "__main__":
    main()
