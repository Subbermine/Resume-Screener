# Resume Screener Frontend (React + Vite)

Glassmorphism UI: upload a resume PDF (click or drag-and-drop), paste a job
description, get ATS + ML scores, matched/missing skill badges, recommendation,
competence/model badge, "Why this score?" explainer badges, and detailed remarks.

## Run

```powershell
npm install
npm run dev     # http://127.0.0.1:5173/
npm run build   # production bundle -> dist/
npm run lint    # oxlint
```

Dev server proxies `/api/*` to the backend at `http://127.0.0.1:8000`
(see `vite.config.js`); start the backend first (`uvicorn app:app` in `backend/`).

## Result fields rendered

`ats_score`, `ml_score` (+ confidence), `skill_score`, `bert_similarity`,
`ensemble_score`, `recommendation`, `decision`, `matched_skills`, `missing_skills`,
`remarks`, `model_version`, `ml_competence`, `top_features`.
