"""Phase 6 inference: raw texts -> Phase-2 features -> learned stacker.

Mirrors training/features.py exactly (same TF-IDF vectorizer, skill IDF,
SBERT model). Single-pair latency is dominated by 2 SBERT encodes (~1s CPU).
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.model_registry import MODEL_VERSION, get_artifacts
from services.education import education_score
from services.experience import experience_score
from services.preprocess import clean_text
from services.scoring import skill_score
from services.skills import SKILLS, extract_skills

try:
    from training.features import FEATURE_COLUMNS
except ImportError:
    from features import FEATURE_COLUMNS  # noqa

TRACK_POOLS = {
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

EDU_RANK = {"none": -1, "bachelor": 0, "master": 1, "phd": 2}
COMPETENCE = {0: "Weak", 1: "Average", 2: "Good", 3: "Strong"}

_SBERT = None


def _sbert():
    global _SBERT
    if _SBERT is None:
        from models.sentence_model import model
        _SBERT = model
    return _SBERT


def _years_resume(text):
    m = re.findall(r"(\d+)\+?\s*years?", text.lower())
    return max(map(int, m)) if m else 0


def _years_req(text):
    m = re.findall(r"(\d+)\+?\s*years?", text.lower())
    return int(m[0]) if m else None


def _edu_level(text):
    t = " " + text.lower() + " "
    if re.search(r"\b(phd|doctorate)\b", t):
        return "phd"
    if re.search(r"\b(master|m\.e\b|m\.tech|m\.sc|mca|mba)\b", t):
        return "master"
    return "bachelor"


def _best_track(skills):
    best, best_n = "other", 0
    for track in TRACK_POOLS:
        n = len(set(skills) & set(TRACK_POOLS[track]))
        if n > best_n:
            best, best_n = track, n
    return best


def build_features(resume_text, jd_text):
    """Return (feature_dict, embedding_row_1152, sbert_cosine, tfidf_cosine)."""
    import numpy as np
    arts = get_artifacts()
    if arts is None:
        raise FileNotFoundError("learned registry unavailable")
    cleaned, job_clean = clean_text(resume_text), clean_text(jd_text)
    rs, js = extract_skills(cleaned), extract_skills(job_clean)
    sk = skill_score(rs, js)
    ex = experience_score(cleaned, job_clean)
    ed = education_score(cleaned, job_clean)
    idf = arts["idf"]
    matched = rs & js
    idf_overlap = sum(idf.get(s, 1.0) for s in matched) / max(1e-9, sum(idf.get(s, 1.0) for s in js)) if js else 0.0
    ry, qy = _years_resume(cleaned), _years_req(job_clean)
    edu_r, edu_j = _edu_level(resume_text), ("none" if re.search(
        r"no strict degree|no degree|regardless of education", jd_text.lower()) else _edu_level(jd_text))
    lw_r, lw_j = math.log1p(len(resume_text.split())), math.log1p(len(jd_text.split()))

    vec = arts["tfidf"].transform([resume_text, jd_text])
    import numpy as _np
    num = float(vec[0].multiply(vec[1]).sum())
    den = float(_np.sqrt(vec[0].multiply(vec[0]).sum()) * _np.sqrt(vec[1].multiply(vec[1]).sum()))
    tfidf = round(max(0.0, num / den if den else 0.0) * 100.0, 2)

    emb = _sbert().encode([resume_text, jd_text], normalize_embeddings=True,
                          convert_to_numpy=True).astype("float32")
    import numpy as np2
    sbert = round(float(max(0.0, min(1.0, float((emb[0] * emb[1]).sum()))) * 100.0), 2)
    diff = np2.abs(emb[0] - emb[1])
    row1152 = np2.concatenate([emb[0], emb[1], diff]).astype("float32")

    fv = {
        "skill_overlap": sk / 100.0, "n_res_skills": len(rs), "n_jd_skills": len(js),
        "matched": len(matched), "missing": len(js - rs),
        "missing_ratio": len(js - rs) / max(1, len(js)),
        "skill_idf_overlap": min(1.0, idf_overlap),
        "exp_score_norm": ex / 100.0,
        "exp_gap_norm": max(-1.0, min(1.0, (ry - (qy or 0)) / 10.0)),
        "exp_meets": int(ex == 100), "edu_match": ed / 100.0,
        "edu_diff": EDU_RANK[edu_r] - EDU_RANK[edu_j],
        "log_len_res": lw_r, "log_len_jd": lw_j,
        "len_ratio": min(lw_r, lw_j) / max(lw_r, lw_j),
        "same_track": int(_best_track(rs) != "other" and _best_track(rs) == _best_track(js)),
        "tfidf_cosine": tfidf, "sbert_cosine": sbert,
        "emb_diff_mean": round(float(diff.mean()), 4), "emb_diff_std": round(float(diff.std()), 4),
    }
    return fv, row1152, sbert, tfidf


def top_features(feature_dict, arts, k=5):
    """Exact linear contributions from the M1 head (scaled x * coef)."""
    import numpy as np
    m1 = arts["m1_logreg.pkl"] if "m1_logreg.pkl" in arts else arts.get("m1")
    scaler, lr = m1.named_steps["standardscaler"], m1.named_steps["logisticregression"]
    x = np.array([[float(feature_dict[c]) for c in FEATURE_COLUMNS]])
    contrib = (scaler.transform(x)[0] * lr.coef_[0])
    order = sorted(range(len(contrib)), key=lambda i: -abs(contrib[i]))[:k]
    names = {"skill_overlap": "skill overlap", "sbert_cosine": "semantic similarity",
             "tfidf_cosine": "keyword overlap", "exp_score_norm": "experience fit",
             "edu_match": "education fit", "missing_ratio": "missing-skill ratio",
             "skill_idf_overlap": "rare-skill overlap", "same_track": "same role track",
             "exp_gap_norm": "experience gap", "matched": "matched skills"}
    return [{"feature": names.get(FEATURE_COLUMNS[i], FEATURE_COLUMNS[i]),
             "contribution": round(float(contrib[i]), 3),
             "direction": "positive" if contrib[i] >= 0 else "negative"} for i in order]


def predict_ml(resume_text, jd_text):
    """Full learned prediction, or None when registry/ML disabled."""
    from services.ensemble import learned_stack_score
    arts = get_artifacts()
    if arts is None:
        return None
    fv, row, sbert, tfidf = build_features(resume_text, jd_text)
    score, level = learned_stack_score(fv, row)
    return {"ml_score": round(score, 2), "ml_confidence": round(score / 100.0, 4),
            "ml_competence": COMPETENCE[level], "ml_competence_level": level,
            "model_version": MODEL_VERSION, "sbert_real": sbert, "tfidf_real": tfidf,
            "top_features": top_features(fv, {"m1": arts["m1_logreg.pkl"]})}
