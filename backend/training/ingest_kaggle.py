"""Ingest Kaggle resume dataset (via HF mirror) + merge with synthetic v1 -> v2.

Source: snehaanbhawal/resume-dataset (CC0, 2482 resumes, livecareer.com scrapes),
mirrored at Divyaamith/Kaggle-Resume (2484 rows, ID/Resume_str/Resume_html/Category).
No Kaggle credentials needed — downloads the public HF parquet mirror (~20MB).

Usage (from backend/):
    python -m training.ingest_kaggle
    python -m training.ingest_kaggle --pairs_per_resume 3 --seed 42 --limit 0

Outputs (v1 files untouched):
    data/raw/kaggle/kaggle_resume.parquet   (normalized cache)
    data/processed/pairs_v2.parquet         (synthetic 3200 + kaggle ~7.4k)
    data/processed/{train,val,test}_v2.parquet (grouped by resume_id, 70/15/15)
    data/processed/stats_v2.json
    data/gold/gold_sample_v2.csv            (200 rows, 100 kaggle + 100 synthetic)

Labels use the same weak ats_proxy family as make_dataset; kaggle Category is
kept as auxiliary column (kaggle_category), never as the label.
"""
import argparse
import json
import os
import random
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

PARQUET_URL = ("https://huggingface.co/datasets/Divyaamith/Kaggle-Resume/"
               "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")
CSV_URL = ("https://huggingface.co/datasets/Divyaamith/Kaggle-Resume/"
           "resolve/main/Resume.csv")

try:
    from training.make_dataset import (
        TRACKS, skill_score_fn, experience_score_fn, education_score_fn,
        _NumpyRngAdapter,
    )
    from training.schema import competence_from_score, COMPETENCE_MAP, MATCH_THRESHOLD
except ImportError:  # allow `python training/ingest_kaggle.py`
    from make_dataset import (
        TRACKS, skill_score_fn, experience_score_fn, education_score_fn,
        _NumpyRngAdapter,
    )
    from schema import competence_from_score, COMPETENCE_MAP, MATCH_THRESHOLD

try:
    from services.skills import extract_skills
except Exception:
    from training.make_dataset import ONTOLOGY as _ONT  # noqa
    def extract_skills(text):
        import re as _re
        t = text.lower()
        return {s for s in _ONT if _re.search(r"\b" + _re.escape(s) + r"\b", t)}


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"cache hit: {dest} ({os.path.getsize(dest)} bytes)")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "resume-screener/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"  {total / 1e6:.1f} MB", end="\r")
    print(f"\nsaved {dest} ({os.path.getsize(dest)} bytes)")
    return dest


def load_kaggle(cache):
    import pandas as pd
    df = pd.read_parquet(cache)
    cols = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    c_id, c_txt, c_cat = col("id"), col("resume_str"), col("category")
    assert c_txt and c_cat, f"unexpected columns: {list(df.columns)}"
    out = pd.DataFrame({
        "kaggle_id": df[c_id].astype(str) if c_id else [f"row{i}" for i in range(len(df))],
        "resume_text": df[c_txt].astype(str),
        "kaggle_category": df[c_cat].astype(str).str.strip().str.upper(),
    })
    out = out[out["resume_text"].str.len() > 100].reset_index(drop=True)
    return out


def extract_exp_years(text):
    years = re.findall(r"(\d+)\+?\s*years?", text.lower())
    return max(map(int, years)) if years else 0


def extract_edu_level(text):
    t = " " + text.lower() + " "
    if re.search(r"\b(phd|doctorate)\b", t):
        return "phd"
    if re.search(r"\b(master|m\.e\b|m\.tech|m\.sc|mca|mba)\b", t):
        return "master"
    if re.search(r"\b(bachelor|b\.e\b|b\.tech|b\.sc|bsc|bca|b\.com|graduate)\b", t):
        return "bachelor"
    return "bachelor"  # livecareer resumes are mostly graduates; neutral default


def best_track(skills, rng):
    counts = {t: len(set(skills) & set(pool)) for t, pool in TRACKS.items()}
    top = max(counts.values())
    if top == 0:
        return None
    cands = [t for t, c in counts.items() if c == top]
    return rng.choice(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_per_resume", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="0 = all resumes")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    adapter = _NumpyRngAdapter(rng, np_rng)

    raw_kaggle = os.path.join(BACKEND, "data", "raw", "kaggle")
    os.makedirs(raw_kaggle, exist_ok=True)
    cache = os.path.join(raw_kaggle, "kaggle_resume.parquet")
    try:
        download(PARQUET_URL, cache)
    except Exception as e:
        print(f"parquet mirror failed ({e}), trying CSV...")
        csv_path = os.path.join(raw_kaggle, "Resume.csv")
        download(CSV_URL, csv_path)
        df_csv = pd.read_csv(csv_path, usecols=lambda c: c.lower() in ("id", "resume_str", "category"))
        df_csv.to_parquet(cache, index=False)

    kag = load_kaggle(cache)
    if args.limit:
        kag = kag.sample(n=min(args.limit, len(kag)), random_state=args.seed).reset_index(drop=True)
    print(f"kaggle resumes: {len(kag)}, categories: {kag.kaggle_category.nunique()}")

    # JD pool: reuse synthetic v1 JDs so JD distribution is identical across sources.
    import json as _json
    jds = []
    with open(os.path.join(BACKEND, "data", "raw", "jds_v1.jsonl"), encoding="utf-8") as f:
        for line in f:
            j = _json.loads(line)
            jds.append({"jd_id": j["jd_id"], "track": j["track"], "text": j["text"],
                        "skills": j["skills"], "req_years": j["req_years"], "req": j["req"]})
    by_track = {}
    for j in jds:
        by_track.setdefault(j["track"], []).append(j)

    pairs = []
    for i, row in kag.iterrows():
        text = row["resume_text"]
        skills = extract_skills(text)
        exp_y = extract_exp_years(text)
        edu = extract_edu_level(text)
        bt = best_track(skills, rng)
        rid = f"K{i:05d}"
        chosen = []
        if bt:
            chosen.append(rng.choice(by_track[bt]))
        while len(chosen) < args.pairs_per_resume:
            jd = rng.choice(jds)
            if jd not in chosen:
                chosen.append(jd)
        for jd in chosen[:args.pairs_per_resume]:
            rs, js = set(skills), set(jd["skills"])
            sk = skill_score_fn(rs, js)
            ex = experience_score_fn(exp_y if exp_y > 0 else None, jd["req_years"])
            ed = education_score_fn(edu, jd["req"])
            overlap = len(rs & js) / max(1, len(js))
            bonus = 8.0 if bt == jd["track"] else 0.0
            bert_proxy = min(100.0, max(0.0, 40.0 + 50.0 * overlap + bonus + float(np_rng.normal(0, 4.0))))
            ats = min(100.0, max(0.0, 0.40 * sk + 0.25 * bert_proxy + 0.20 * ex + 0.15 * ed))
            level = competence_from_score(ats)
            pairs.append({
                "pair_id": f"PK{len(pairs):06d}", "resume_id": rid, "jd_id": jd["jd_id"],
                "track_resume": bt or "other", "track_jd": jd["track"],
                "resume_text": text, "jd_text": jd["text"],
                "resume_skills": ";".join(sorted(rs)), "jd_skills": ";".join(sorted(js)),
                "exp_years_resume": exp_y, "exp_years_req": jd["req_years"],
                "edu_resume": edu, "edu_req": jd["req"],
                "skill_score": round(sk, 2), "experience_score": round(ex, 2),
                "education_score": round(ed, 2), "bert_proxy": round(bert_proxy, 2),
                "ats_proxy": round(ats, 2),
                "label_match": int(ats >= MATCH_THRESHOLD),
                "competence_level": level, "competence_label": COMPETENCE_MAP[level],
                "split": "unassigned", "source": "kaggle",
                "kaggle_category": row["kaggle_category"],
            })

    kdf = pd.DataFrame(pairs)
    syn = pd.read_parquet(os.path.join(BACKEND, "data", "processed", "pairs_v1.parquet")).copy()
    syn["source"] = "synthetic"
    syn["kaggle_category"] = ""
    full = pd.concat([syn, kdf], ignore_index=True)

    from sklearn.model_selection import GroupShuffleSplit
    groups = full["resume_id"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=args.seed)
    tr, te = next(gss.split(full, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed + 1)
    rv, rt = next(gss2.split(te, groups=groups[te]))
    full.loc[full.index[tr], "split"] = "train"
    full.loc[full.index[te[rv]], "split"] = "val"
    full.loc[full.index[te[rt]], "split"] = "test"

    proc = os.path.join(BACKEND, "data", "processed")
    full.to_parquet(os.path.join(proc, "pairs_v2.parquet"), index=False)
    for s in ["train", "val", "test"]:
        full[full["split"] == s].to_parquet(os.path.join(proc, f"{s}_v2.parquet"), index=False)

    stats = {
        "n_pairs": len(full), "n_kaggle_pairs": len(kdf), "n_synthetic_pairs": len(syn),
        "n_kaggle_resumes": int(kag["kaggle_id"].nunique()),
        "seed": args.seed, "pairs_per_resume_kaggle": args.pairs_per_resume,
        "match_rate_overall": round(float(full["label_match"].mean()), 4),
        "match_rate_by_source": {s: round(float(full[full.source == s]["label_match"].mean()), 4)
                                 for s in ["synthetic", "kaggle"]},
        "match_rate_by_split": {s: round(float(full[full.split == s]["label_match"].mean()), 4)
                                for s in ["train", "val", "test"]},
        "competence_dist": full["competence_label"].value_counts().to_dict(),
        "split_counts": full["split"].value_counts().to_dict(),
        "top_kaggle_categories": kag["kaggle_category"].value_counts().head(10).to_dict(),
        "sources": ["snehaanbhawal/resume-dataset (CC0) via Divyaamith/Kaggle-Resume HF mirror",
                    "synthetic v1 (templates x skills ontology)"],
        "note": "Weak labels; kaggle Category is auxiliary only, never the label.",
    }
    with open(os.path.join(proc, "stats_v2.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    gold_k = full[full.source == "kaggle"].sample(n=min(100, (full.source == "kaggle").sum()), random_state=args.seed)
    gold_s = full[full.source == "synthetic"].sample(n=min(100, (full.source == "synthetic").sum()), random_state=args.seed + 1)
    gold = pd.concat([gold_k, gold_s]).sample(frac=1, random_state=args.seed)
    gold["reviewer_label_match"] = ""
    gold["reviewer_competence"] = ""
    gold["reviewer_notes"] = ""
    gold.to_csv(os.path.join(BACKEND, "data", "gold", "gold_sample_v2.csv"), index=False)

    print(json.dumps(stats, indent=2))
    print(f"Wrote {len(full)} v2 pairs -> {proc}")


if __name__ == "__main__":
    main()
