# Resume Screener — ML-Based Screening with Sentence-BERT + Learned Ensemble

Upload a resume PDF + job description, get an explainable evaluation:
heuristic ATS score plus a calibrated **ML score** (`ensemble_v1`) with confidence,
competence level, and "why this score" feature badges. Batch-rank up to 10 resumes via `/rank`.

## Quickstart

```powershell
# backend (from backend/)
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000

# frontend (from frontend/)
npm install
npm run dev      # http://127.0.0.1:5173/  (proxies /api -> :8000)
```

First backend start downloads `all-MiniLM-L6-v2` (~90MB, cached after).

## API

`POST /analyze` — form fields `resume` (PDF), `job_description` (text).
Returns legacy fields (`ats_score`, `skill_score`, `bert_similarity`, `experience_score`,
`education_score`, `ensemble_score`, `matched_skills`, `missing_skills`, `recommendation`,
`decision`, `remarks`) plus `ml_score`, `ml_confidence`, `ml_competence`,
`model_version` (`ensemble_v1`, or `heuristic` when ML is off/unavailable), `top_features[5]`.
Set `USE_ML_MODEL=0` for heuristic-only mode.

`POST /rank` — form fields `resumes` (up to 10 PDFs), `job_description`; returns them
ranked by `ml_score` (falls back to `ats_score`) with `rank` per item.

## Scores

* Heuristic ATS: `0.40*skills + 0.25*SBERT-cosine + 0.20*experience + 0.15*education`.
* Learned stacker: LogReg + HistGradientBoosting + MLP-on-embeddings (Level-0, 5-fold OOF),
  LogReg meta, Platt calibration, threshold 0.45. Test: acc 0.996, F1 0.987, ECE 0.015
  (heuristic ECE 0.305). Competence head (Weak/Average/Good/Strong): acc 0.987.
* Caveat: labels are weak (`ats_proxy>=60`), so near-1.0 AUCs are rule recovery —
  recalibrate on human gold before production use.

## ML pipeline (all commands from `backend/`)

```
python -m training.make_dataset --n_resumes 800 --pairs_per_resume 4 --seed 42   # synthetic v1 (3200 pairs)
python -m training.ingest_kaggle --pairs_per_resume 3 --seed 42                  # + Kaggle real resumes -> v2 (10649)
python -m training.features --version both                                       # 20 features + real SBERT + TF-IDF (train-fit)
python -m training.train_baselines --version 2                                   # M1/M2/M3 -> registry
python -m training.train_ensemble --version 2                                    # stacking + calibration + competence head
python -m training.evaluate_fairness --version 2                                 # slice report -> fairness_v2.json
python -m training.validate_dataset                                              # v1+v2 checks
python -m unittest discover -s tests                                             # 35 tests
```

Details, metrics, and limitations: `data/README.md`. Artifacts: `models/registry/ensemble_v1/`.

## Human gold labels (Fix 4)

Fill `reviewer_label_match` (0/1) and `reviewer_competence` (0-3) in
`data/gold/gold_sample_v2.csv` (200 rows), then:

```
python -m training.gold status
python -m training.gold agreement     # weak-vs-human kappa/F1
python -m training.gold recalibrate   # gold threshold + Platt -> ensemble_v1_gold/
```

## Transformer fine-tune M4 (GPU, RTX 3060 12GB)

Needs CUDA torch (`pip install torch --index-url https://download.pytorch.org/whl/cu126`):

```powershell
python -m training.train_transformer --run_full --epochs 3 --sample 0 --max_length 256 --batch_size 32 --fp16
```

Saves `models/registry/ensemble_v1/transformer_m4/` (~5-10 min). CPU smoke:
`python -m training.train_transformer --smoke --sample 32 --max_length 64`.

## Paper reference

[Competence-Level Prediction and Resume & Job Description Matching Using Context-Aware
Transformer Models](https://arxiv.org/abs/2011.02998) trains section-aware classifiers on
labeled data. This repo bootstraps from weak labels (pretrained SBERT + heuristics), adds a
real-resume Kaggle slice (CC0), and is built to improve once gold labels land.
