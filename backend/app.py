import os

from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form

from services.parser import extract_pdf
from services.preprocess import clean_text
from services.matcher import bert_similarity
from services.skills import extract_skills
from services.scoring import skill_score
from services.experience import experience_score
from services.education import education_score
from services.remarks import generate_remark
from services.ensemble import boosted_ensemble_score

app = FastAPI(title="Resume Screener (heuristic + learned ensemble_v1)")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
MAX_RANK_FILES = 10

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _score_pair(resume_text, job_description):
    """Shared heuristic + learned scoring for one (resume, JD) pair."""
    cleaned = clean_text(resume_text)
    job_clean = clean_text(job_description)

    if not cleaned:
        raise HTTPException(status_code=422, detail="No readable text was found in the uploaded PDF.")
    if not job_clean:
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    resume_skills = extract_skills(cleaned)
    job_skills = extract_skills(job_clean)
    bert = bert_similarity(cleaned, job_clean)
    skills = skill_score(resume_skills, job_skills)
    experience = experience_score(cleaned, job_clean)
    education = education_score(cleaned, job_clean)

    final_score = 0.40 * skills + 0.25 * bert + 0.20 * experience + 0.15 * education
    ensemble_score = boosted_ensemble_score(bert, skills, experience, education)
    remark = generate_remark(final_score, resume_skills.intersection(job_skills),
                             job_skills - resume_skills, experience, education)

    # Learned path: additive, never breaks the legacy fields on failure.
    ml = None
    try:
        from services.ml_inference import predict_ml
        ml = predict_ml(resume_text, job_description)
    except Exception:
        ml = None

    result = {
        "bert_similarity": round(bert, 2),
        "skill_score": round(skills, 2),
        "experience_score": round(experience, 2),
        "education_score": round(education, 2),
        "ats_score": round(final_score, 2),
        "ensemble_score": round(ensemble_score, 2),
        "matched_skills": list(resume_skills.intersection(job_skills)),
        "missing_skills": list(job_skills - resume_skills),
        "recommendation": remark["recommendation"],
        "decision": remark["decision"],
        "remarks": remark["remarks"],
    }
    if ml is None:
        result.update({"model_version": "heuristic", "ml_score": None,
                       "ml_confidence": None, "ml_competence": None,
                       "top_features": []})
    else:
        result.update({"model_version": ml["model_version"], "ml_score": ml["ml_score"],
                       "ml_confidence": ml["ml_confidence"], "ml_competence": ml["ml_competence"],
                       "ml_competence_level": ml["ml_competence_level"],
                       "top_features": ml["top_features"]})
    return result


@app.post("/analyze")
async def analyze_resume(
        resume: UploadFile = File(...),
        job_description: str = Form(...)
):

    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    path = os.path.join(UPLOAD_FOLDER, os.path.basename(resume.filename))

    with open(path, "wb") as f:
        f.write(await resume.read())

    text = extract_pdf(path)
    return _score_pair(text, job_description)


@app.post("/rank")
async def rank_resumes(
        resumes: List[UploadFile] = File(...),
        job_description: str = Form(...)
):
    """Batch-screen up to MAX_RANK_FILES PDFs, ranked by ml_score (or ats_score)."""
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    if len(resumes) > MAX_RANK_FILES:
        raise HTTPException(status_code=400, detail=f"Max {MAX_RANK_FILES} resumes per request.")

    ranked = []
    for resume in resumes:
        if not resume.filename or not resume.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF resumes are supported ({resume.filename}).")
        path = os.path.join(UPLOAD_FOLDER, os.path.basename(resume.filename))
        with open(path, "wb") as f:
            f.write(await resume.read())
        scored = _score_pair(extract_pdf(path), job_description)
        scored["filename"] = resume.filename
        ranked.append(scored)

    key = lambda r: (r["ml_score"] if r["ml_score"] is not None else r["ats_score"])
    ranked.sort(key=key, reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return {"count": len(ranked), "rank_by": "ml_score", "results": ranked}