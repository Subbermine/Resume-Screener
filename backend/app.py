import os

from fastapi import FastAPI, HTTPException, UploadFile, File, Form

from services.parser import extract_pdf
from services.preprocess import clean_text
from services.matcher import bert_similarity
from services.skills import extract_skills
from services.scoring import skill_score
from services.experience import experience_score
from services.education import education_score
from services.remarks import generate_remark

app = FastAPI()

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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
    cleaned = clean_text(text)
    job_clean = clean_text(job_description)

    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="No readable text was found in the uploaded PDF."
        )

    if not job_clean:
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    resume_skills = extract_skills(cleaned)
    job_skills = extract_skills(job_clean)

    bert = bert_similarity(cleaned, job_clean)

    skills = skill_score(
        resume_skills,
        job_skills
    )

    experience = experience_score(
        cleaned,
        job_clean
    )

    education = education_score(
        cleaned,
        job_clean
    )

    final_score = (
       0.40 * skills +
0.25 * bert +
0.20 * experience +
0.15 * education
    )

    remark = generate_remark(
    final_score,
    resume_skills.intersection(job_skills),
    job_skills - resume_skills,
    experience,
    education
)

    return {
    "bert_similarity": round(bert, 2),
    "skill_score": round(skills, 2),
    "experience_score": round(experience, 2),
    "education_score": round(education, 2),
    "ats_score": round(final_score, 2),

    "matched_skills": list(resume_skills.intersection(job_skills)),
    "missing_skills": list(job_skills - resume_skills),

    "recommendation": remark["recommendation"],
    "decision": remark["decision"],
    
    "remarks": remark["remarks"]
}