"""Dataset v1 schema for resume-JD matching.

One row = one (resume x JD) pair. Labels are weak/bootstrapping labels
derived from the same heuristic family as backend/app.py so Phase 2/3 can
train a learned ensemble to replace the hand-coded weights.
Human gold review happens on data/gold/gold_sample_v1.csv.
"""

PAIR_COLUMNS = [
    "pair_id",
    "resume_id",
    "jd_id",
    "track_resume",
    "track_jd",
    "resume_text",
    "jd_text",
    "resume_skills",      # sorted list stored as ;-joined string in parquet/csv
    "jd_skills",
    "exp_years_resume",
    "exp_years_req",
    "edu_resume",         # bachelor | master | phd
    "edu_req",            # bachelor | master | phd | none
    "skill_score",        # 0-100
    "experience_score",   # 0-100
    "education_score",    # 0/100
    "bert_proxy",         # 0-100, SBERT stand-in (Phase 2 computes real cosine)
    "ats_proxy",          # 0-100 weighted like app.py
    "label_match",        # 0/1, 1 if ats_proxy >= 60
    "competence_level",   # 0 Weak | 1 Average | 2 Good | 3 Strong
    "competence_label",   # string name
    "split",              # train | val | test
]

COMPETENCE_MAP = {0: "Weak", 1: "Average", 2: "Good", 3: "Strong"}

# v2 adds provenance columns (kaggle merge). v1 files are untouched.
PAIR_COLUMNS_V2 = PAIR_COLUMNS + ["source", "kaggle_category"]

# Thresholds mirror backend/services/remarks.py recommendation tiers.
MATCH_THRESHOLD = 60.0
COMPETENCE_THRESHOLDS = [(85.0, 3), (70.0, 2), (50.0, 1)]  # else 0


def competence_from_score(score: float) -> int:
    for thresh, level in COMPETENCE_THRESHOLDS:
        if score >= thresh:
            return level
    return 0
