# Dataset v1 — Resume x JD matching (weak labels, Phase 1)

3200 pairs from 800 synthetic resumes x 200 JDs, 4 pairs/resume.
Regenerate: `python -m training.make_dataset --n_resumes 800 --pairs_per_resume 4 --seed 42`
Validate: `python -m training.validate_dataset`

## Files
- `raw/resumes_v1.jsonl` — resume_id, track, text, skills, exp_years, edu
- `raw/jds_v1.jsonl` — jd_id, track, text, skills, req_years, req
- `processed/pairs_v1.parquet` — full labeled set (see `training/schema.py`)
- `processed/{train,val,test}_v1.parquet` — 2240/480/480, grouped by resume_id (no leakage)
- `processed/stats_v1.json` — match_rate 0.567, competence {Average:1260, Good:763, Weak:762, Strong:415}
- `gold/gold_sample_v1.csv` — 200 rows, fill reviewer_label_match / reviewer_competence / reviewer_notes

## Labels (weak, bootstrapping only)
`ats_proxy = 0.40*skill + 0.25*bert_proxy + 0.20*exp + 0.15*edu`, same weights as `app.py`.
`bert_proxy = 40 + 50*skill_overlap + track_bonus(8 if same track) + N(0,4)`, clipped 0-100.
`label_match = ats_proxy >= 60`; competence `>=85 Strong, >=70 Good, >=50 Average, else Weak` (mirrors `services/remarks.py`).
Phase 2 replaces `bert_proxy` with real SBERT cosine as a training feature; gold review provides human ground truth.

## Limitations
Synthetic phrasing, ~40-skill ontology from `services/skills.py`, naive years/degree text.
Do not use for hiring decisions. Human-review gold sample before training production models.

---

# Dataset v2 — v1 + Kaggle real resumes (weak labels)

Regenerate: `python -m training.ingest_kaggle --pairs_per_resume 3 --seed 42`
Validate: `python -m training.validate_dataset --version 2`

## Source
`snehaanbhawal/resume-dataset` (CC0 Public Domain, 2482 livecareer.com resumes, 24 categories)
via public HF mirror `Divyaamith/Kaggle-Resume` (`ID, Resume_str, Resume_html, Category`).
No Kaggle credentials needed. Cached at `raw/kaggle/kaggle_resume.parquet` (~20MB).
JDs reused from v1 (`raw/jds_v1.jsonl`, 200 tech JDs) so JD distribution is identical across sources.

## Files (v1 untouched)
- `processed/pairs_v2.parquet` — **10649 pairs** (3200 synthetic + 7449 kaggle), schema = v1 + `source, kaggle_category`
- `processed/{train,val,test}_v2.parquet` — 7446/1606/1597, grouped by resume_id (no leakage, synthetic/kaggle resume_ids disjoint)
- `processed/stats_v2.json`, `gold/gold_sample_v2.csv` (100 kaggle + 100 synthetic for review)

## Label reality (read before training)
- Overall match rate 0.175; synthetic slice 0.568, kaggle slice 0.006.
- Why: 69% of Kaggle resumes (HR/chef/teacher/...) contain zero tech-ontology skills, so vs tech JDs they are
  legitimately negatives — realistic hard negatives with real resume phrasing. Positives come from the
  synthetic slice + IT/Engineering resumes (238). `kaggle_category` is auxiliary only, never the label.
- Train with `class_weight='balanced'` (or sampling) + threshold tuning; see stats_v2.json.
- Known pre-existing hazard: `\bc\b` in `services/skills.py` fires on any standalone "c" (inflates IT skill counts slightly).

## License note
Kaggle slice is CC0; synthetic slice is project-generated. Keep this attribution when redistributing v2.

---

# Phase 2 — Features (structured + TF-IDF + real SBERT)

Run: `python -m training.features --version both` (v1 ~1 min, v2 ~2 min on CPU, `all-MiniLM-L6-v2`).

## Artifacts per version
- `features_v{v}.parquet` — 1 row/pair: meta + 20 features + `label_match/competence_level` + reference `bert_proxy/ats_proxy`
- `tfidf_v{v}.pkl` — TfidfVectorizer(10k, 1-2gram, sublinear), fit on TRAIN split only
- `feature_config_v{v}.json`, `feature_stats_v{v}.json`
- `embeddings_v{v}.npz` — unique resume/JD SBERT embeddings (deduped: v1 799+200, v2 3280+200) for Phase 3

## Features (20)
`skill_overlap, n_res_skills, n_jd_skills, matched, missing, missing_ratio, skill_idf_overlap (train-fit IDF),`
`exp_score_norm, exp_gap_norm, exp_meets, edu_match, edu_diff, log_len_res, log_len_jd, len_ratio, same_track,`
`tfidf_cosine, sbert_cosine, emb_diff_mean, emb_diff_std`.

## Validation (measured)
- v1: sbert-vs-proxy corr 0.62, tfidf-vs-sbert 0.54 — real SBERT adds signal beyond the synthetic proxy.
- v2: sbert mean 34.3 (vs v1 67.7), consistent with cross-domain negatives; corr 0.78/0.78 (signals agree on floor).
- Tests: `tests/test_features.py` (9 tests) — row parity with pairs, no NaNs, cosine ranges, train-only vectorizer.

---

# Phases 3+4 — Base models + learned stacking ensemble (weak labels)

Run: `python -m training.train_baselines --version 2` then `python -m training.train_ensemble --version 2`.
M4 (optional, GPU): `python -m training.train_transformer --run_full` (CPU `--smoke` verified: trains 1 epoch end-to-end).
RTX 3060 12GB command (needs CUDA torch first, this box is CPU-only torch):
`pip install torch --index-url https://download.pytorch.org/whl/cu126`, then from `backend/`:
`python -m training.train_transformer --run_full --epochs 3 --sample 0 --max_length 256 --batch_size 32 --fp16`
(~5-10 min, saves `models/registry/ensemble_v1/transformer_m4/`).

## Models
- M1 LogReg on 20 compact features | M2 HistGradientBoosting on 20 features | M3 MLP on 1152-d SBERT `[R,J,|R-J|]`
- Meta: LogReg on `[m1,m2,m3,sbert,skill_overlap]` (5-fold OOF) + Platt scaling fit on val; tuned threshold 0.45
- Competence head: LogReg 4-class (Weak/Average/Good/Strong). Registry: `models/registry/ensemble_v1/` + `config.json`

## Test results (v2 holdout, n=1597) — read the caveat
| model | acc | f1 | AUC | ECE |
|---|---|---|---|---|
| heuristic ats_proxy | 1.000 | 1.000 | 1.000 | 0.305 |
| M1 | 0.991 | 0.974 | 1.000 | 0.013 |
| M2 HGB | 0.997 | 0.991 | 1.000 | 0.004 |
| M3 MLP-emb | 0.949 | 0.842 | 0.985 | 0.018 |
| stack calibrated | 0.996 | 0.987 | 1.000 | 0.015 |
| competence head | acc 0.987, macro-F1 0.964 (majority 0.758) | | | |
| ablation AUC | full 0.9999; w/o skills 0.9925, w/o experience 0.9923, w/o education 0.9969, w/o semantic 0.9998 |

Caveat: labels derive from `ats_proxy>=60`, so any model seeing the same signals recovers the rule (~1.0 AUC is
recovery, not intelligence). Real wins vs heuristic: calibrated probabilities (ECE 0.015 vs 0.305), learned
ranking platform, competence head. Recalibrate on human gold before production use.
Services: `services/ensemble.py` keeps `boosted_ensemble_score` unchanged + adds `is_learned_available()` /
`learned_stack_score(feature_dict, embedding_row)` (app.py wiring = Phase 6).

---

# Phases 5-7 — Fairness, backend integration, frontend

- Fairness: `python -m training.evaluate_fairness --version 2` → `processed/fairness_v2.json`.
  v2 test flags: kaggle slice sel-rate 0.003 (expected: real non-tech negatives), phd reqs 0.098,
  frontend/backend/data_ml ±0.04 — all F1/AUC ≈ 1.0 (consistent, weak-label ceiling). Revisit on gold labels.
- API: `/analyze` keeps all legacy fields + adds `ml_score, ml_confidence, ml_competence,
  model_version, top_features` (exact linear M1 contributions). `USE_ML_MODEL=0` → heuristic fallback.
  New `POST /rank` (≤10 PDFs, ranked by ml_score). Shared logic in `app._score_pair`.
- Inference: `services/ml_inference.py` rebuilds the 20 Phase-2 features live (train-fit TF-IDF/IDF
  loaded from disk, SBERT reused); `models/model_registry.py` lazy-loads `ensemble_v1`.
- Frontend: ML score card + confidence, competence + model badge, "Why this score?" badges. `npm run build` OK.
- Tests: `tests/test_api.py` (5: happy path incl. legacy-math check, non-PDF reject, rank order, rank cap, ML-off).
