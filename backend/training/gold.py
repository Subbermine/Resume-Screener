"""Fix 4: gold human-label workflow — status, agreement, recalibration.

The 200-row gold sample (data/gold/gold_sample_v2.csv, 100 kaggle + 100
synthetic) ships with empty reviewer columns. Fill them in Excel/Sheets:
  reviewer_label_match: 0 or 1 (is this resume a match for the JD?)
  reviewer_competence:  0-3 or Weak/Average/Good/Strong
  reviewer_notes:       free text (optional)

Then:
  python -m training.gold status        # fill progress
  python -m training.gold agreement     # weak-vs-human agreement (kappa, F1)
  python -m training.gold recalibrate   # tune threshold + refit Platt on gold
                                        # -> models/registry/ensemble_v1_gold/
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

GOLD = os.path.join(BACKEND, "data", "gold", "gold_sample_v2.csv")
COMP = {"weak": 0, "average": 1, "good": 2, "strong": 3}


def load_gold():
    import pandas as pd
    df = pd.read_csv(GOLD)
    lab = df["reviewer_label_match"]
    filled = lab.astype(str).str.strip().str.lower().isin(["0", "1", "0.0", "1.0", "true", "false"])
    g = df[filled].copy()
    g["gold_label"] = lab[filled].astype(str).str.strip().str.lower().map(
        lambda v: 1 if v in ("1", "1.0", "true") else 0)
    comp = g["reviewer_competence"].astype(str).str.strip().str.lower()
    g["gold_level"] = comp.map(lambda v: COMP.get(v, int(float(v)) if v.replace(".", "").isdigit() else -1))
    g = g[g["gold_level"].isin([0, 1, 2, 3])]
    return df, g


def gold_scores(g):
    """Calibrated stacker probabilities for gold pairs via the registry."""
    import pickle
    import numpy as np
    import pandas as pd
    from training.features import FEATURE_COLUMNS
    from training.utils import REGISTRY, pair_embedding_matrix
    feat = pd.read_parquet(os.path.join(BACKEND, "data", "processed", "features_v2.parquet"))
    f = feat[feat["pair_id"].isin(set(g["pair_id"]))].set_index("pair_id").loc[g["pair_id"]].reset_index()
    X = f[FEATURE_COLUMNS].to_numpy()
    E = pair_embedding_matrix("2", f)

    def load(n):
        with open(os.path.join(REGISTRY, n), "rb") as fh:
            return pickle.load(fh)
    m1, m2, m3, sc3, meta, cal = (load(n) for n in
        ["m1_logreg.pkl", "hgb_m2.pkl", "mlp_m3.pkl", "scaler_m3.pkl",
         "meta_logreg.pkl", "calibrator.pkl"])
    F = np.column_stack([m1.predict_proba(X)[:, 1], m2.predict_proba(X)[:, 1],
                         m3.predict_proba(sc3.transform(E))[:, 1],
                         f["sbert_cosine"].to_numpy() / 100.0,
                         f["skill_overlap"].to_numpy()])
    return cal.predict_proba(meta.predict_proba(F)[:, 1].reshape(-1, 1))[:, 1]


def cmd_status():
    df, g = load_gold()
    print(f"gold filled: {len(g)}/{len(df)}")
    if "source" in df.columns:
        print("by source:", df.assign(f= df["reviewer_label_match"].astype(str).str.strip() != "").groupby("source")["f"].sum().to_dict())
    if not len(g):
        print("Fill reviewer_label_match/reviewer_competence in data/gold/gold_sample_v2.csv first.")
        sys.exit(2)


def cmd_agreement():
    import numpy as np
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
    df, g = load_gold()
    if not len(g):
        sys.exit("no filled gold rows — fill the CSV first (see module docstring)")
    p = gold_scores(g)
    pred = (p >= 0.45).astype(int)
    print(f"n={len(g)} weak-vs-gold: acc={accuracy_score(g['gold_label'], pred):.3f} "
          f"f1={f1_score(g['gold_label'], pred):.3f} kappa={cohen_kappa_score(g['gold_label'], pred):.3f}")
    print("confusion [[tn fp][fn tp]]:", confusion_matrix(g["gold_label"], pred).tolist())
    print(f"weak label_match rate: {g['label_match'].mean():.3f} | gold rate: {g['gold_label'].mean():.3f}")


def cmd_recalibrate():
    import json
    import pickle
    import numpy as np
    from sklearn.metrics import f1_score
    df, g = load_gold()
    if len(g) < 20:
        sys.exit(f"need >=20 filled gold rows, have {len(g)}")
    p = gold_scores(g)
    y = g["gold_label"].to_numpy()
    thrs = np.round(np.arange(0.2, 0.81, 0.05), 2)
    tuned = max(thrs, key=lambda t: f1_score(y, (p >= t).astype(int)))
    from sklearn.linear_model import LogisticRegression
    cal_gold = LogisticRegression(max_iter=1000, random_state=42).fit(p.reshape(-1, 1), y)
    out = os.path.join(BACKEND, "models", "registry", "ensemble_v1_gold")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "calibrator_gold.pkl"), "wb") as f:
        pickle.dump(cal_gold, f)
    pg = cal_gold.predict_proba(p.reshape(-1, 1))[:, 1]
    metrics = {"n_gold": len(g), "tuned_threshold_gold": float(tuned),
               "f1_at_tuned": round(float(f1_score(y, (pg >= tuned).astype(int))), 4),
               "base": "ensemble_v1 (frozen); only threshold + Platt refit on gold"}
    json.dump(metrics, open(os.path.join(out, "metrics_gold.json"), "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    print("saved ->", out)
    print("To serve gold calibration, point model_registry at ensemble_v1_gold (Phase 6 follow-up).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"status": cmd_status, "agreement": cmd_agreement, "recalibrate": cmd_recalibrate}[cmd]()
