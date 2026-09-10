"""Phase 4: learned stacking ensemble + calibration + competence head.

Level-0 (5-fold OOF on train): M1 LogReg, M2 HGB, M3 MLP-embeddings.
Meta-features: [m1, m2, m3, sbert_cosine/100, skill_overlap].
Meta: LogisticRegression (class_weight balanced) + sigmoid calibration (prefit on val).
Competence head: multinomial LogReg on compact features (Weak/Average/Good/Strong).
Ablation: LogReg w/o each signal group -> test AUC delta.

Usage (from backend/): python -m training.train_ensemble [--version 2]
Requires: python -m training.train_baselines first (saved full-train L0 models).
Saves: meta_logreg.pkl, calibrator.pkl, competence_head.pkl, oof_preds.npz,
       metrics_ensemble.json, config.json into models/registry/ensemble_v1/.

NOTE on near-perfect AUCs: labels are weak labels derived from ats_proxy
(threshold 60), so models that see the same signals recover the rule almost
exactly. The learned stacker's value vs the heuristic is (a) calibrated
probabilities (heuristic ECE ~0.30), (b) ranking robustness, (c) a platform
that improves once human gold labels replace/augment weak labels.
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
    from training.utils import (ABLATION_GROUPS, REGISTRY, binary_metrics,
                                ece_score, load_features, make_m1, make_m2,
                                make_m3, pair_embedding_matrix, precision_at_k,
                                save_json)
except ImportError:
    from features import FEATURE_COLUMNS
    from utils import (ABLATION_GROUPS, REGISTRY, binary_metrics, ece_score,
                       load_features, make_m1, make_m2, make_m3,
                       pair_embedding_matrix, precision_at_k, save_json)

META_COLS = ["m1", "m2", "m3", "sbert", "skill_overlap"]


def oof_matrix(X, E, y, n_splits=5, seed=42):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros((len(X), 3))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(X, y):
        oof[va_idx, 0] = make_m1().fit(X[tr_idx], y[tr_idx]).predict_proba(X[va_idx])[:, 1]
        oof[va_idx, 1] = make_m2().fit(X[tr_idx], y[tr_idx]).predict_proba(X[va_idx])[:, 1]
        sc = StandardScaler().fit(E[tr_idx])
        oof[va_idx, 2] = make_m3().fit(sc.transform(E[tr_idx]), y[tr_idx]).predict_proba(sc.transform(E[va_idx]))[:, 1]
    return oof


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="2", choices=["1", "2"])
    args = ap.parse_args()
    t0 = time.time()

    for f in ["m1_logreg.pkl", "hgb_m2.pkl", "mlp_m3.pkl", "scaler_m3.pkl"]:
        if not os.path.exists(os.path.join(REGISTRY, f)):
            sys.exit(f"missing {f}: run `python -m training.train_baselines --version {args.version}` first")

    df, tr, va, te = load_features(args.version)
    Xtr, ytr = tr[FEATURE_COLUMNS].to_numpy(), tr["label_match"].to_numpy()
    Xva, yva = va[FEATURE_COLUMNS].to_numpy(), va["label_match"].to_numpy()
    Xte, yte = te[FEATURE_COLUMNS].to_numpy(), te["label_match"].to_numpy()
    Etr = pair_embedding_matrix(args.version, tr)
    Ete = pair_embedding_matrix(args.version, te)

    print("building 5-fold OOF on train...")
    oof = oof_matrix(Xtr, Etr, ytr)
    Ftr = np.column_stack([oof, tr["sbert_cosine"].to_numpy() / 100.0, tr["skill_overlap"].to_numpy()])

    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    meta.fit(Ftr, ytr)

    # Full-train L0 (saved by baselines) -> val/test meta-features.
    with open(os.path.join(REGISTRY, "m1_logreg.pkl"), "rb") as f:
        m1 = pickle.load(f)
    with open(os.path.join(REGISTRY, "hgb_m2.pkl"), "rb") as f:
        m2 = pickle.load(f)
    with open(os.path.join(REGISTRY, "mlp_m3.pkl"), "rb") as f:
        m3 = pickle.load(f)
    with open(os.path.join(REGISTRY, "scaler_m3.pkl"), "rb") as f:
        sc3 = pickle.load(f)
    Eva = pair_embedding_matrix(args.version, va)

    def meta_features(X, E, split):
        return np.column_stack([m1.predict_proba(X)[:, 1], m2.predict_proba(X)[:, 1],
                                m3.predict_proba(sc3.transform(E))[:, 1],
                                split["sbert_cosine"].to_numpy() / 100.0,
                                split["skill_overlap"].to_numpy()])

    Fva, Fte = meta_features(Xva, Eva, va), meta_features(Xte, Ete, te)
    p_uncal_va, p_uncal_te = meta.predict_proba(Fva)[:, 1], meta.predict_proba(Fte)[:, 1]

    # Platt (sigmoid) calibration: LogReg on val stacker scores (sklearn 1.9
    # removed CalibratedClassifierCV(cv="prefit"); this is the same model).
    cal = LogisticRegression(max_iter=1000, random_state=42)
    cal.fit(p_uncal_va.reshape(-1, 1), yva)
    pva = cal.predict_proba(p_uncal_va.reshape(-1, 1))[:, 1]
    pte = cal.predict_proba(p_uncal_te.reshape(-1, 1))[:, 1]

    # Threshold tuned on val (max F1), reported on test.
    thrs = np.round(np.arange(0.2, 0.81, 0.05), 2)
    from sklearn.metrics import f1_score
    tuned = max(thrs, key=lambda t: f1_score(yva, (pva >= t).astype(int)))

    out = {"version": args.version, "meta_features": META_COLS,
           "tuned_threshold_val_f1": float(tuned), "models": {}}
    for name, prob in [("stack_uncalibrated", p_uncal_te), ("stack_calibrated", pte)]:
        m = {**binary_metrics(yte, prob, (prob >= 0.5).astype(int)),
             **dict(zip(["p_at_5", "p_at_5_groups"], precision_at_k(te.assign(score=prob), "score")))}
        out["models"][name] = m
        print(name, m)
    mt = {**binary_metrics(yte, pte, (pte >= tuned).astype(int)),
          **dict(zip(["p_at_5", "p_at_5_groups"], precision_at_k(te.assign(score=pte), "score")))}
    out["models"][f"stack_calibrated_t{tuned}"] = mt
    print("tuned", mt)
    out["models"]["calibration_gain_ece"] = round(ece_score(yte, p_uncal_te) - ece_score(yte, pte), 4)

    # Competence head (multiclass) on compact features.
    from sklearn.metrics import accuracy_score, f1_score as _f1
    from sklearn.preprocessing import StandardScaler
    yc_tr, yc_te = tr["competence_level"].to_numpy(), te["competence_level"].to_numpy()
    sc_c = StandardScaler().fit(Xtr)
    comp = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    comp.fit(sc_c.transform(Xtr), yc_tr)
    pred_c = comp.predict(sc_c.transform(Xte))
    import collections
    majority = collections.Counter(yc_te).most_common(1)[0][1] / len(yc_te)
    out["models"]["competence_head"] = {
        "acc": round(float(accuracy_score(yc_te, pred_c)), 4),
        "macro_f1": round(float(_f1(yc_te, pred_c, average="macro")), 4),
        "majority_acc": round(float(majority), 4),
    }
    print("competence", out["models"]["competence_head"])

    # Ablation: LogReg without each signal group.
    out["ablation_auc"] = {}
    from sklearn.metrics import roc_auc_score
    full_auc = roc_auc_score(yte, make_m1().fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    out["ablation_auc"]["full_m1"] = round(float(full_auc), 4)
    for group, cols in ABLATION_GROUPS.items():
        keep = [c for c in FEATURE_COLUMNS if c not in cols]
        idx = [FEATURE_COLUMNS.index(c) for c in keep]
        pa = make_m1().fit(Xtr[:, idx], ytr).predict_proba(Xte[:, idx])[:, 1]
        out["ablation_auc"][f"without_{group}"] = round(float(roc_auc_score(yte, pa)), 4)
    print("ablation", out["ablation_auc"])

    with open(os.path.join(REGISTRY, "meta_logreg.pkl"), "wb") as f:
        pickle.dump(meta, f)
    with open(os.path.join(REGISTRY, "calibrator.pkl"), "wb") as f:
        pickle.dump(cal, f)
    with open(os.path.join(REGISTRY, "competence_head.pkl"), "wb") as f:
        pickle.dump((sc_c, comp), f)
    np.savez_compressed(os.path.join(REGISTRY, "oof_preds.npz"), oof=oof, y_train=ytr)
    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, os.path.join(REGISTRY, "metrics_ensemble.json"))
    save_json({"model_version": "ensemble_v1", "trained_on": f"features_v{args.version}",
               "seed": 42, "n_folds": 5, "meta_features": META_COLS,
               "tuned_threshold": float(tuned),
               "files": ["m1_logreg.pkl", "hgb_m2.pkl", "mlp_m3.pkl", "scaler_m3.pkl",
                         "meta_logreg.pkl", "calibrator.pkl", "competence_head.pkl"],
               "targets": {"match": "label_match>=threshold", "competence": "competence_level 0-3"},
               "note": "Weak-label training; recalibrate on human gold before production use."},
              os.path.join(REGISTRY, "config.json"))
    print(f"saved -> {REGISTRY} ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
