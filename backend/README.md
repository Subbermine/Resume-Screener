# Resume Screener Backend

This API combines three signals:

- Transformer semantic similarity: `all-MiniLM-L6-v2` creates embeddings for the cleaned resume and job description. Their cosine similarity is returned as a 0-100 semantic score.
- Skill overlap: extracted skills are compared as sets.
- Heuristic requirements: experience and education are scored separately.

The ATS score is weighted as 40% skills, 25% semantic similarity, 20% experience, and 15% education.

## Run locally

From this directory:

```powershell
python -m pip install -r requirements.txt
uvicorn app:app --reload
```

The frontend proxies `/api/analyze` to `http://127.0.0.1:8000/analyze`.

## Paper reference

The referenced paper, [Competence-Level Prediction and Resume & Job Description Matching Using Context-Aware Transformer Models](https://arxiv.org/abs/2011.02998), trains section-aware transformer classifiers with labeled resume data. This project does not have that labeled training dataset, so it uses a pretrained sentence-transformer embedding model plus cosine similarity for an explainable screening baseline. The skill, experience, and education signals provide structured evidence alongside the semantic score.
