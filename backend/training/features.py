"""Phase 2: feature pipeline — structured + TF-IDF + real SBERT cosine.

Usage (from backend/):
    python -m training.features --version 1        # fast sanity (~3k pairs)
    python -m training.features --version 2        # full (~10.6k pairs)
    python -m training.features --version both     # default
    python -m training.features --version 2 --no_sbert   # skip embeddings

Leakage discipline:
- TF-IDF vectorizer + skill IDF are fit on TRAIN split only, then applied
  to val/test. SBERT is pretrained (no fitting) so all texts can be encoded.
- Speedup: encode UNIQUE resume/JD texts once, then map back to pairs.

Outputs per version v in processed/:
    features_v{v}.parquet       (one row per pair: meta + 20 features + targets)
    tfidf_v{v}.pkl              (fitted vectorizer, train-only)
    feature_config_v{v}.json    (params, model name, sbert_available, IDF size)
    embeddings_v{v}.npz         (unique resume/JD SBERT embeddings for Phase 3)
    feature_stats_v{v}.json     (means/stds, sbert-vs-proxy correlation)

Features (20): skill_overlap, n_res_skills, n_jd_skills, matched, missing,
missing_ratio, skill_idf_overlap, exp_score_norm, exp_gap_norm, exp_meets,
edu_match, edu_diff, log_len_res, log_len_jd, len_ratio, same_track,
tfidf_cosine, sbert_cosine, emb_diff_mean, emb_diff_std.
Reference-only (NOT training features): bert_proxy, ats_proxy.
"""
import argparse
import json
import math
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

MODEL_NAME = "all-MiniLM-L6-v2"

FEATURE_COLUMNS = [
    "skill_overlap", "n_res_skills", "n_jd_skills", "matched", "missing",
    "missing_ratio", "skill_idf_overlap", "exp_score_norm", "exp_gap_norm",
    "exp_meets", "edu_match", "edu_diff", "log_len_res", "log_len_jd",
    "len_ratio", "same_track", "tfidf_cosine", "sbert_cosine",
    "emb_diff_mean", "emb_diff_std",
]

EDU_RANK = {"none": -1, "bachelor": 0, "master": 1, "phd": 2}


def parse_skills(s):
    return set(x for x in str(s).split(";") if x) if str(s) else set()


def structured_rows(df, idf):
    rows = []
    for _, r in df.iterrows():
        rs, js = parse_skills(r["resume_skills"]), parse_skills(r["jd_skills"])
        matched = rs & js
        missing_ratio = len(js - rs) / max(1, len(js))
        idf_overlap = (sum(idf.get(s, 1.0) for s in matched)
                       / max(1e-9, sum(idf.get(s, 1.0) for s in js)))
        exp_gap = (r["exp_years_resume"] - r["exp_years_req"]) / 10.0
        edu_diff = EDU_RANK.get(r["edu_resume"], 0) - EDU_RANK.get(r["edu_req"], -1)
        lw_r = math.log1p(len(str(r["resume_text"]).split()))
        lw_j = math.log1p(len(str(r["jd_text"]).split()))
        rows.append({
            "pair_id": r["pair_id"],
            "skill_overlap": r["skill_score"] / 100.0,
            "n_res_skills": len(rs), "n_jd_skills": len(js),
            "matched": len(matched), "missing": len(js - rs),
            "missing_ratio": missing_ratio,
            "skill_idf_overlap": min(1.0, idf_overlap),
            "exp_score_norm": r["experience_score"] / 100.0,
            "exp_gap_norm": max(-1.0, min(1.0, exp_gap)),
            "exp_meets": int(r["experience_score"] == 100),
            "edu_match": r["education_score"] / 100.0,
            "edu_diff": edu_diff,
            "log_len_res": lw_r, "log_len_jd": lw_j,
            "len_ratio": min(lw_r, lw_j) / max(lw_r, lw_j),
            "same_track": int(r["track_resume"] == r["track_jd"]),
        })
    import pandas as pd
    return pd.DataFrame(rows).set_index("pair_id")


def fit_idf(train_df):
    import numpy as np
    docs = [parse_skills(s) for s in train_df["resume_skills"]]
    n = max(1, len(docs))
    df_count = {}
    for d in docs:
        for s in d:
            df_count[s] = df_count.get(s, 0) + 1
    return {s: float(np.log((1 + n) / (1 + c)) + 1.0) for s, c in df_count.items()}


def tfidf_cosine(train_texts, all_resume, all_jd):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                          sublinear_tf=True, lowercase=True)
    vec.fit(train_texts)
    R = vec.transform(all_resume)
    J = vec.transform(all_jd)
    num = np.array(R.multiply(J).sum(axis=1)).ravel()
    denom = np.sqrt(np.array(R.multiply(R).sum(axis=1)).ravel()) * \
        np.sqrt(np.array(J.multiply(J).sum(axis=1)).ravel())
    cos = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
    return vec, np.clip(cos, 0, 1) * 100.0


def sbert_encode(unique_texts, batch_size):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(list(unique_texts), batch_size=batch_size,
                       normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=True)
    return np.asarray(emb, dtype="float32")


def build_version(proc, v, batch_size, use_sbert):
    import numpy as np
    import pandas as pd
    df = pd.read_parquet(os.path.join(proc, f"pairs_v{v}.parquet"))
    train = df[df["split"] == "train"]
    print(f"[v{v}] pairs={len(df)} train={len(train)}")

    idf = fit_idf(train)
    struct = structured_rows(df, idf)

    train_texts = pd.concat([train["resume_text"], train["jd_text"]]).astype(str).tolist()
    vec, tfidf = tfidf_cosine(train_texts, df["resume_text"].astype(str).tolist(),
                              df["jd_text"].astype(str).tolist())
    with open(os.path.join(proc, f"tfidf_v{v}.pkl"), "wb") as f:
        pickle.dump(vec, f)

    if use_sbert:
        res_texts = df["resume_text"].astype(str)
        jd_texts = df["jd_text"].astype(str)
        uniq_r = pd.unique(res_texts)
        uniq_j = pd.unique(jd_texts)
        print(f"[v{v}] encoding {len(uniq_r)} unique resumes + {len(uniq_j)} unique JDs")
        Er = sbert_encode(uniq_r, batch_size)
        Ej = sbert_encode(uniq_j, batch_size)
        r_index = {t: k for k, t in enumerate(uniq_r)}
        j_index = {t: k for k, t in enumerate(uniq_j)}
        ri = res_texts.map(r_index).to_numpy()
        ji = jd_texts.map(j_index).to_numpy()
        cos = np.clip((Er[ri] * Ej[ji]).sum(axis=1), -1, 1) * 100.0
        cos = np.clip(cos, 0, 100)
        diff = np.abs(Er[ri] - Ej[ji])
        dmean, dstd = diff.mean(axis=1), diff.std(axis=1)
        np.savez_compressed(os.path.join(proc, f"embeddings_v{v}.npz"),
                            resume_ids=np.array([f"R{i}" for i in range(len(uniq_r))]),
                            jd_ids=np.array([f"J{i}" for i in range(len(uniq_j))]),
                            resume_texts=np.array(uniq_r, dtype=object),
                            jd_texts=np.array(uniq_j, dtype=object),
                            resume_emb=Er, jd_emb=Ej)
        sbert_available = True
    else:
        cos = df["bert_proxy"].to_numpy(dtype=float)
        dmean = np.zeros(len(df))
        dstd = np.zeros(len(df))
        sbert_available = False

    feat = struct.copy()
    feat["tfidf_cosine"] = np.round(tfidf, 2)
    feat["sbert_cosine"] = np.round(cos, 2)
    feat["emb_diff_mean"] = np.round(dmean, 4)
    feat["emb_diff_std"] = np.round(dstd, 4)
    feat = feat.reset_index()

    meta = df[["pair_id", "resume_id", "jd_id", "split", "label_match",
               "competence_level", "bert_proxy", "ats_proxy"] +
              (["source", "kaggle_category"] if "source" in df.columns else [])]
    out = meta.merge(feat, on="pair_id", how="left")
    assert len(out) == len(df) and out[FEATURE_COLUMNS].isna().sum().sum() == 0
    out.to_parquet(os.path.join(proc, f"features_v{v}.parquet"), index=False)

    stats = {
        "n": len(out),
        "sbert_available": sbert_available,
        "model": MODEL_NAME if sbert_available else None,
        "means": {c: round(float(out[c].mean()), 4) for c in FEATURE_COLUMNS},
        "sbert_vs_proxy_corr": round(float(out[["sbert_cosine", "bert_proxy"]].corr().iloc[0, 1]), 4),
        "tfidf_vs_sbert_corr": round(float(out[["tfidf_cosine", "sbert_cosine"]].corr().iloc[0, 1]), 4),
        "match_rate": round(float(out["label_match"].mean()), 4),
    }
    with open(os.path.join(proc, f"feature_stats_v{v}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(proc, f"feature_config_v{v}.json"), "w") as f:
        json.dump({"model_name": MODEL_NAME, "sbert_available": sbert_available,
                   "tfidf": {"max_features": 10000, "ngram_range": [1, 2], "sublinear_tf": True,
                             "fit_on": "train_split_only"},
                   "idf": {"fit_on": "train_resume_skills", "n_terms": len(idf)},
                   "features": FEATURE_COLUMNS,
                   "reference_only": ["bert_proxy", "ats_proxy"]}, f, indent=2)
    print(f"[v{v}] " + json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="both", choices=["1", "2", "both"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--no_sbert", action="store_true")
    args = ap.parse_args()
    proc = os.path.join(BACKEND, "data", "processed")
    versions = ["1", "2"] if args.version == "both" else [args.version]
    for v in versions:
        build_version(proc, v, args.batch_size, not args.no_sbert)


if __name__ == "__main__":
    main()
