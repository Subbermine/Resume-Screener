def skill_score(resume_skills, job_skills):
    if len(job_skills) == 0:
        return 0  # or return None and handle it separately

    matched = resume_skills.intersection(job_skills)
    return (len(matched) / len(job_skills)) * 100