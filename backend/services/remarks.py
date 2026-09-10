def generate_remark(
    ats_score,
    matched_skills,
    missing_skills,
    experience_score,
    education_score
):
    # Recommendation
    if ats_score >= 85:
        recommendation = "Strong Match"
    elif ats_score >= 70:
        recommendation = "Good Match"
    elif ats_score >= 50:
        recommendation = "Average Match"
    else:
        recommendation = "Weak Match"

    remarks = []

    # Overall assessment
    if ats_score >= 85:
        remarks.append(
            "The candidate is highly suitable for the position based on technical skills, experience, and overall profile."
        )
    elif ats_score >= 70:
        remarks.append(
            "The candidate meets most of the job requirements and is a good fit for the role."
        )
    elif ats_score >= 50:
        remarks.append(
            "The candidate satisfies some of the requirements but has noticeable skill gaps."
        )
    else:
        remarks.append(
            "The candidate does not currently meet the essential requirements for this role."
        )

    # Skills
    if matched_skills:
        remarks.append(
            f"Matched {len(matched_skills)} required technical skills."
        )

    if missing_skills:
        remarks.append(
            f"Missing {len(missing_skills)} important skill(s): {', '.join(sorted(missing_skills))}."
        )
    else:
        remarks.append(
            "No required technical skills are missing."
        )

    # Experience
    if experience_score == 100:
        remarks.append(
            "Experience satisfies the specified job requirements."
        )
    else:
        remarks.append(
            "Relevant work experience could be improved."
        )

    # Education
    if education_score == 100:
        remarks.append(
            "Educational qualifications satisfy the required criteria."
        )
    else:
        remarks.append(
            "Educational qualifications could not be fully verified against the job requirements."
        )

    decision = generate_decision(ats_score)

    return {
        "recommendation": recommendation,
        "remarks": " ".join(remarks),
        "decision":decision
    }

def generate_decision(score):
    if score >= 90:
        return "Highly Recommended"

    elif score >= 75:
        return "Shortlist for Interview"

    elif score >= 60:
        return "Consider for Interview"

    elif score >= 40:
        return "Hold for Future Consideration"

    return "Not Recommended"