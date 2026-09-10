"""Build dataset v1: synthetic resumes + JDs + weak labels + grouped splits.

Usage (from backend/):
    python -m training.make_dataset --n_resumes 800 --pairs_per_resume 4 --seed 42
    python training/make_dataset.py --n_resumes 800 --pairs_per_resume 4 --seed 42

Outputs under backend/data/:
    raw/resumes_v1.jsonl, raw/jds_v1.jsonl
    processed/pairs_v1.parquet, train_v1.parquet, val_v1.parquet, test_v1.parquet
    processed/stats_v1.json
    gold/gold_sample_v1.csv  (200 rows for human review)

Design notes:
- Ontology reused from services/skills.py (single source of truth).
- Scoring reuses services/scoring.py, experience.py, education.py logic via
  local copies so dataset generation never imports torch/sentence-transformers.
- bert_proxy is a documented stand-in: 40 + 50*skill_overlap + noise, clipped.
  Phase 2 replaces it with real SBERT cosine as a training feature. Labels are
  therefore bootstrapping/weak labels, NOT human ground truth. Gold sample
  captures the human-verification step.
- Splits are grouped by resume_id (no resume leaks across train/val/test).
"""
import argparse
import json
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

try:
    from services.skills import SKILLS as ONTOLOGY
except Exception:  # fallback if run from odd cwd
    ONTOLOGY = {
        "python", "java", "c", "javascript", "typescript", "html", "css",
        "react", "angular", "vue", "node", "express", "mongodb", "mysql",
        "postgresql", "sql", "docker", "kubernetes", "aws", "azure", "gcp",
        "git", "github", "linux", "spring", "spring boot", "fastapi",
        "flask", "django", "tensorflow", "pytorch", "machine learning",
        "deep learning", "pandas", "numpy", "scikit-learn",
    }

try:
    from training.schema import competence_from_score, COMPETENCE_MAP, MATCH_THRESHOLD
except ImportError:  # allow `python training/make_dataset.py` from backend/
    from schema import competence_from_score, COMPETENCE_MAP, MATCH_THRESHOLD

TRACKS = {
    "backend": ["python", "java", "sql", "postgresql", "mysql", "mongodb",
                "django", "flask", "fastapi", "spring", "spring boot",
                "docker", "git", "linux", "aws"],
    "frontend": ["javascript", "typescript", "html", "css", "react", "angular",
                 "vue", "node", "express", "git", "github"],
    "data_ml": ["python", "sql", "pandas", "numpy", "scikit-learn",
                "tensorflow", "pytorch", "machine learning", "deep learning",
                "mysql", "postgresql", "git", "aws", "docker"],
    "devops": ["docker", "kubernetes", "aws", "azure", "gcp", "linux",
               "git", "github", "python", "sql"],
    "fullstack": ["javascript", "typescript", "react", "node", "express",
                  "python", "sql", "mongodb", "docker", "git", "aws"],
}

DEGREES = [
    ("B.Tech in Computer Science", "bachelor"),
    ("Bachelor of Science in Information Technology", "bachelor"),
    ("BCA in Computer Applications", "bachelor"),
    ("M.Tech in Software Engineering", "master"),
    ("Master of Science in Data Science", "master"),
    ("MCA in Computer Applications", "master"),
    ("PhD in Machine Learning", "phd"),
]

EDU_REQ_OPTIONS = ["bachelor", "bachelor", "bachelor", "master", "master", "none", "phd"]
EDU_REQ_TEXT = {
    "bachelor": "Bachelor's degree in Computer Science or related field required.",
    "master": "Master's degree in Computer Science or related field required.",
    "phd": "PhD or doctorate preferred.",
    "none": "Relevant practical experience valued; no strict degree requirement.",
}

SUMMARY_TEMPLATES = [
    "Software engineer with {y} years of experience building scalable applications.",
    "Results-driven developer with {y} years of hands-on industry experience.",
    "Motivated engineer with {y} years of experience across full project lifecycles.",
]
EXP_TEMPLATES = [
    "Worked for {y} years as a software engineer delivering production systems.",
    "Total of {y} years experience in software development and team collaboration.",
    "{y} years of professional experience including design, testing and deployment.",
]
FRESHER_TEXT = "Recent graduate seeking entry-level opportunity with strong project work."


def skill_score_fn(resume_skills, job_skills):
    if not job_skills:
        return 0.0
    return len(set(resume_skills) & set(job_skills)) / len(job_skills) * 100.0


def experience_score_fn(res_years, req_years):
    if req_years is None:
        return 100.0
    if res_years is None or res_years <= 0:
        return 0.0
    if res_years >= req_years:
        return 100.0
    return res_years / req_years * 100.0


def education_score_fn(res_level, req_level):
    rank = {"none": -1, "bachelor": 0, "master": 1, "phd": 2}
    if req_level == "none" or req_level is None:
        return 100.0
    if res_level is None:
        return 0.0
    return 100.0 if rank[res_level] >= rank[req_level] else 0.0


def make_resume_text(rng, skills, exp_years, degree_text):
    skill_str = ", ".join(sorted(skills))
    summary = rng.choice(SUMMARY_TEMPLATES).format(y=exp_years) if exp_years > 0 else FRESHER_TEXT
    exp = rng.choice(EXP_TEMPLATES).format(y=exp_years) if exp_years > 0 else "Academic projects and internships covering design and implementation."
    return (
        f"Summary: {summary}\n"
        f"Technical Skills: {skill_str}.\n"
        f"Experience: {exp}\n"
        f"Education: {degree_text}.\n"
    )


def make_jd_text(rng, title, skills, req_years, req_level):
    skill_str = ", ".join(sorted(skills))
    exp_line = f"Minimum {req_years} years of relevant experience required." if req_years else "Open to candidates at various experience levels."
    return (
        f"Job Title: {title}\n"
        f"Requirements: Strong background in {skill_str}.\n"
        f"Experience: {exp_line}\n"
        f"Education: {EDU_REQ_TEXT[req_level]}\n"
    )


def gen_resumes(rng, n):
    resumes = []
    tracks = list(TRACKS)
    for i in range(n):
        track = tracks[i % len(tracks)]
        pool = TRACKS[track]
        k = rng.randint(5, min(9, len(pool)))
        skills = set(rng.sample(pool, k))
        # sprinkle 0-2 generic skills
        for g in rng.sample(["git", "linux", "sql"], rng.randint(0, 2)):
            if g in ONTOLOGY:
                skills.add(g)
        skills = {s for s in skills if s in ONTOLOGY} or set(rng.sample(pool, 3))
        exp_years = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], p=[0.12, 0.14, 0.16, 0.16, 0.14, 0.12, 0.08, 0.05, 0.03]))
        deg_text, deg_level = DEGREES[rng.integers(0, len(DEGREES))]
        text = make_resume_text(rng, skills, exp_years, deg_text)
        resumes.append({"resume_id": f"R{i:05d}", "track": track, "text": text,
                        "skills": sorted(skills), "exp_years": exp_years,
                        "edu": deg_level, "edu_text": deg_text})
    return resumes


def gen_jds(rng, n):
    titles = {"backend": "Backend Developer", "frontend": "Frontend Developer",
              "data_ml": "ML Engineer", "devops": "DevOps Engineer",
              "fullstack": "Full Stack Developer"}
    tracks = list(TRACKS)
    jds = []
    for i in range(n):
        track = tracks[i % len(tracks)]
        pool = TRACKS[track]
        k = rng.randint(4, min(7, len(pool)))
        skills = set(rng.sample(pool, k))
        skills = {s for s in skills if s in ONTOLOGY}
        req_years = int(rng.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.25, 0.15, 0.1]))
        req_level = rng.choice(EDU_REQ_OPTIONS)
        text = make_jd_text(rng, titles[track], skills, req_years, req_level)
        jds.append({"jd_id": f"J{i:05d}", "track": track, "text": text,
                    "skills": sorted(skills), "req_years": req_years, "req": req_level})
    return jds


def pair_and_label(rng, resumes, jds_by_track, all_jds, pairs_per_resume):
    pairs = []
    pid = 0
    for r in resumes:
        jd_pool_same = jds_by_track[r["track"]]
        # 1) positive-leaning: same track, bias JD skills toward resume skills
        jd_same = rng.choice(jd_pool_same)
        # 2) hard negative: same track but force low overlap by picking JD then swapping skills if overlap high
        jd_hard = rng.choice(jd_pool_same)
        # 3..n) random tracks
        chosen = [jd_same, jd_hard]
        while len(chosen) < pairs_per_resume:
            chosen.append(rng.choice(all_jds))
        for jd in chosen[:pairs_per_resume]:
            rs, js = set(r["skills"]), set(jd["skills"])
            sk = skill_score_fn(rs, js)
            ex = experience_score_fn(r["exp_years"], jd["req_years"])
            ed = education_score_fn(r["edu"], jd["req"])
            overlap = len(rs & js) / max(1, len(js))
            track_bonus = 8.0 if r["track"] == jd["track"] else 0.0
            noise = float(rng.normal(0, 4.0))
            bert_proxy = min(100.0, max(0.0, 40.0 + 50.0 * overlap + track_bonus + noise))
            ats = 0.40 * sk + 0.25 * bert_proxy + 0.20 * ex + 0.15 * ed
            ats = min(100.0, max(0.0, ats))
            level = competence_from_score(ats)
            pairs.append({
                "pair_id": f"P{pid:06d}", "resume_id": r["resume_id"], "jd_id": jd["jd_id"],
                "track_resume": r["track"], "track_jd": jd["track"],
                "resume_text": r["text"], "jd_text": jd["text"],
                "resume_skills": ";".join(sorted(rs)), "jd_skills": ";".join(sorted(js)),
                "exp_years_resume": r["exp_years"], "exp_years_req": jd["req_years"],
                "edu_resume": r["edu"], "edu_req": jd["req"],
                "skill_score": round(sk, 2), "experience_score": round(ex, 2),
                "education_score": round(ed, 2), "bert_proxy": round(bert_proxy, 2),
                "ats_proxy": round(ats, 2),
                "label_match": int(ats >= MATCH_THRESHOLD),
                "competence_level": level, "competence_label": COMPETENCE_MAP[level],
                "split": "unassigned",
            })
            pid += 1
    return pairs


def grouped_split(pairs, rng_seed):
    # Group by resume_id: 70/15/15 without leakage.
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit
    df = pd.DataFrame(pairs)
    groups = df["resume_id"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=rng_seed)
    train_idx, temp_idx = next(gss.split(df, groups=groups))
    temp_groups = groups[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=rng_seed + 1)
    # apply on temp subset with its own group labels
    rel_val, rel_test = next(gss2.split(temp_idx, groups=temp_groups))
    val_idx = temp_idx[rel_val]
    test_idx = temp_idx[rel_test]
    df.loc[df.index[train_idx], "split"] = "train"
    df.loc[df.index[val_idx], "split"] = "val"
    df.loc[df.index[test_idx], "split"] = "test"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_resumes", type=int, default=800)
    ap.add_argument("--n_jds", type=int, default=200)
    ap.add_argument("--pairs_per_resume", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gold_n", type=int, default=200)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    # use numpy Generator-backed random wrapper for choice/sample consistency
    class RNG:
        def __init__(self, r, npr):
            self.r = r
            self.npr = npr
        def choice(self, seq, p=None):
            if p is not None:
                return str(self.npr.choice(list(seq), p=np.array(p)))
            return self.r.choice(list(seq))
        def sample(self, seq, k):
            return self.r.sample(list(seq), k)
        def randint(self, a, b):
            return self.r.randint(a, b)
        def integers(self, a, b):
            return int(self.npr.integers(a, b))

    rngw = RNG(rng, np_rng)

    resumes = gen_resumes(rngw, args.n_resumes)
    jds = gen_jds(rngw, args.n_jds)
    jds_by_track = {}
    for jd in jds:
        jds_by_track.setdefault(jd["track"], []).append(jd)

    # attach numpy rng for gaussian noise inside pairing via monkey attr
    pairs = pair_and_label(_NumpyRngAdapter(rng, np_rng), resumes, jds_by_track, jds, args.pairs_per_resume)
    df = grouped_split(pairs, args.seed)

    data_dir = os.path.join(BACKEND, "data")
    raw_dir = os.path.join(data_dir, "raw")
    proc_dir = os.path.join(data_dir, "processed")
    gold_dir = os.path.join(data_dir, "gold")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)

    with open(os.path.join(raw_dir, "resumes_v1.jsonl"), "w", encoding="utf-8") as f:
        for r in resumes:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(raw_dir, "jds_v1.jsonl"), "w", encoding="utf-8") as f:
        for j in jds:
            f.write(json.dumps(j) + "\n")

    df.to_parquet(os.path.join(proc_dir, "pairs_v1.parquet"), index=False)
    for split in ["train", "val", "test"]:
        df[df["split"] == split].to_parquet(os.path.join(proc_dir, f"{split}_v1.parquet"), index=False)

    stats = {
        "n_resumes": len(resumes), "n_jds": len(jds), "n_pairs": len(df),
        "seed": args.seed,
        "match_rate_overall": round(float(df["label_match"].mean()), 4),
        "match_rate_by_split": {s: round(float(df[df["split"] == s]["label_match"].mean()), 4) for s in ["train", "val", "test"]},
        "competence_dist": df["competence_label"].value_counts().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "note": "Weak/bootstrapping labels. bert_proxy is synthetic; Phase 2 replaces with real SBERT cosine.",
    }
    with open(os.path.join(proc_dir, "stats_v1.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    gold = df.sample(n=min(args.gold_n, len(df)), random_state=args.seed).copy()
    gold["reviewer_label_match"] = ""
    gold["reviewer_competence"] = ""
    gold["reviewer_notes"] = ""
    gold.to_csv(os.path.join(gold_dir, "gold_sample_v1.csv"), index=False)

    print(json.dumps(stats, indent=2))
    print(f"Wrote {len(df)} pairs -> {proc_dir}")
    print(f"Gold sample -> {gold_dir}/gold_sample_v1.csv")


class _NumpyRngAdapter:
    """Minimal adapter exposing choice/sample/randint/normal used by pairing."""
    def __init__(self, rng, npr):
        self.rng = rng
        self.npr = npr
    def choice(self, seq):
        return self.rng.choice(list(seq))
    def sample(self, seq, k):
        return self.rng.sample(list(seq), k)
    def randint(self, a, b):
        return self.rng.randint(a, b)
    def normal(self, mu, sigma):
        return float(self.npr.normal(mu, sigma))


if __name__ == "__main__":
    main()
