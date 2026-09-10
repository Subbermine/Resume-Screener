"""Model registry: lazy loading of ensemble_v1 artifacts + feature plumbing.

USE_ML_MODEL=0 disables the learned path (heuristic-only mode).
"""
import json
import os
import pickle

MODEL_VERSION = "ensemble_v1"

REGISTRY_DIR = os.path.join(os.path.dirname(__file__), "registry", MODEL_VERSION)
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

_CACHE = {}


def ml_enabled():
    return os.environ.get("USE_ML_MODEL", "1") != "0"


def _load_pickle(name):
    if name not in _CACHE:
        path = os.path.join(REGISTRY_DIR, name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            _CACHE[name] = pickle.load(f)
    return _CACHE[name]


def _load_json(name):
    if name not in _CACHE:
        for base in (REGISTRY_DIR, PROCESSED_DIR):
            path = os.path.join(base, name)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    _CACHE[name] = json.load(f)
                break
        else:
            return None
    return _CACHE[name]


def get_artifacts():
    """Return dict of all learned artifacts, or None if registry incomplete."""
    if not ml_enabled():
        return None
    needed = ["m1_logreg.pkl", "hgb_m2.pkl", "mlp_m3.pkl", "scaler_m3.pkl",
              "meta_logreg.pkl", "calibrator.pkl", "competence_head.pkl"]
    arts = {n: _load_pickle(n) for n in needed}
    arts["tfidf"] = _load_pickle("tfidf_v2.pkl") or _load_pickle_processed("tfidf_v2.pkl")
    arts["idf"] = _load_json("skill_idf_v2.json")
    arts["config"] = _load_json("config.json")
    if any(v is None for v in arts.values()):
        return None
    return arts


def _load_pickle_processed(name):
    key = "proc:" + name
    if key not in _CACHE:
        path = os.path.join(PROCESSED_DIR, name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            _CACHE[key] = pickle.load(f)
    return _CACHE[key]
