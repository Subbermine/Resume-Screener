def clamp(value):
    return max(0.0, min(100.0, value))


def boosted_ensemble_score(semantic_similarity, skill_score, experience_score, education_score):
    """
    Score a resume using a simple ensemble and boosting layer.

    This project does not have a labeled training dataset, so the file
    implements an inference-time stacking heuristic over the four available
    evidence sources:
      - semantic similarity (BERT embeddings + cosine)
      - skills overlap ratio
      - experience ratio
      - education gate

    The weights mirror the project’s existing ATS blend while allowing a
    small data-driven style boost when all signals agree.
    """
    # Keep the same weight profile the API already uses.
    base = (
        0.40 * clamp(skill_score) +
        0.25 * clamp(semantic_similarity) +
        0.20 * clamp(experience_score) +
        0.15 * clamp(education_score)
    )

    # Boost when multiple high-confidence feature signals agree.
    boost = 0.0

    if skill_score >= 80 and semantic_similarity >= 75:
        boost += 4.0

    if experience_score >= 75 and education_score == 100:
        boost += 2.0

    if skill_score >= 70 and semantic_similarity >= 70 and experience_score >= 70:
        boost += 3.0

    if skill_score < 30 and semantic_similarity < 30:
        boost -= 2.5

    # A mild penalty when the semantic gap is high but the hard skills are weak.
    if semantic_similarity >= 70 and skill_score < 40:
        boost -= 2.0

    return clamp(base + boost)
