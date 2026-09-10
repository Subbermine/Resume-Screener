import re

def experience_score(resume_text, job_description):

    resume = resume_text.lower()
    job = job_description.lower()

    job_years = re.findall(r"(\d+)\+?\s*years?", job)

    if not job_years:
        return 100

    required = int(job_years[0])

    resume_years = re.findall(r"(\d+)\+?\s*years?", resume)

    if not resume_years:
        return 0

    candidate = max(map(int, resume_years))

    if candidate >= required:
        return 100

    return (candidate / required) * 100