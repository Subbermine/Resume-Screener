"""Heuristic ensemble (legacy) + learned stacking ensemble (Phase 4).

- boosted_ensemble_score: original hand-coded blend, unchanged (fallback).
- Learned path: models/registry/ensemble_v1/ (LogReg + HGB + MLP-embeddings
  Level-0, LogReg meta, Platt calibration). It needs the full 20-dim feature
  vector from training/features.py, NOT just the four scalars, so inference
  passes a feature dict. app.py wiring is Phase 6; this module only exposes
  is_learned_available() + learned_stack_score().
"""
import os
import pickle

_REGISTRY = os.path.join(os.path.dirname(__file__), "..", "models", "registry", "ensemble_v1")
_CACHE = {}


def clamp(value):
    return max(0.0, min(100.0, value))


def boosted_ensemble_score(semantic_similarity, skill_score, experience_score, education_score):
    """
    Score a resume using a simple ensemble and boosting layer.

    This project does not have a labeled training dataset, so the file
    implements an inference-time stacking heuristic over the four available
    evidence sources:
      - semantic similarity (BERT embeddings + cosine)
      - skills overlap ratio
      - experience ratio
      - education gate

    The weights mirror the project’s existing ATS blend while allowing a
    small data-driven style boost when all signals agree.
    """
    # Keep the same weight profile the API already uses.
    base = (
        0.40 * clamp(skill_score) +
        0.25 * clamp(semantic_similarity) +
        0.20 * clamp(experience_score) +
        0.15 * clamp(education_score)
    )

    # Boost when multiple high-confidence feature signals agree.
    boost = 0.0

    if skill_score >= 80 and semantic_similarity >= 75:
        boost += 4.0

    if experience_score >= 75 and education_score == 100:
        boost += 2.0

    if skill_score >= 70 and semantic_similarity >= 70 and experience_score >= 70:
        boost += 3.0

    if skill_score < 30 and semantic_similarity < 30:
        boost -= 2.5

    # A mild penalty when the semantic gap is high but the hard skills are weak.
    if semantic_similarity >= 70 and skill_score < 40:
        boost -= 2.0

    return clamp(base + boost)


# ---------------------------------------------------------------- learned path

META_COLS = ["m1", "m2", "m3", "sbert", "skill_overlap"]


def _load(name):
    if name not in _CACHE:
        path = os.path.join(_REGISTRY, name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            _CACHE[name] = pickle.load(f)
    return _CACHE[name]


def is_learned_available():
    return all(
        os.path.exists(os.path.join(_REGISTRY, f))
        for f in ["m1_logreg.pkl", "hgb_m2.pkl", "mlp_m3.pkl", "scaler_m3.pkl",
                  "meta_logreg.pkl", "calibrator.pkl", "competence_head.pkl", "config.json"]
    )


def learned_stack_score(feature_vector, embedding_vector=None):
    """Score a pair from the Phase-2 feature dict + optional 1152-d embedding row.

    feature_vector: dict with the 20 FEATURE_COLUMNS keys.
    embedding_vector: array-like of dim 1152 ([R, J, |R-J|]); required for the
        M3 MLP head. If None, M3 falls back to sbert_cosine/100 as its signal.
    Returns (calibrated_proba_0_100, competence_level_0_3).
    """
    import numpy as np

    try:
        from training.features import FEATURE_COLUMNS
    except ImportError:
        from features import FEATURE_COLUMNS  # noqa

    m1 = _load("m1_logreg.pkl")
    m2 = _load("hgb_m2.pkl")
    m3 = _load("mlp_m3.pkl")
    sc3 = _load("scaler_m3.pkl")
    meta = _load("meta_logreg.pkl")
    cal = _load("calibrator.pkl")
    sc_c, comp = _load("competence_head.pkl")
    if None in (m1, m2, m3, sc3, meta, cal, sc_c, comp):
        raise FileNotFoundError("learned registry incomplete")

    x = np.array([[float(feature_vector[c]) for c in FEATURE_COLUMNS]])
    if embedding_vector is None:
        p3 = float(feature_vector.get("sbert_cosine", 0.0)) / 100.0
    else:
        p3 = float(m3.predict_proba(sc3.transform(np.asarray(embedding_vector).reshape(1, -1)))[:, 1][0])
    fmeta = np.array([[
        float(m1.predict_proba(x)[:, 1][0]),
        float(m2.predict_proba(x)[:, 1][0]),
        p3,
        float(feature_vector.get("sbert_cosine", 0.0)) / 100.0,
        float(feature_vector.get("skill_overlap", 0.0)),
    ]])
    uncal = float(meta.predict_proba(fmeta)[:, 1][0])
    proba = float(cal.predict_proba(np.array([[uncal]]))[:, 1][0])
    level = int(comp.predict(sc_c.transform(x))[0])
    return clamp(proba * 100.0), level
