from sklearn.metrics.pairwise import cosine_similarity
from models.sentence_model import model

def bert_similarity(resume, job):
    if not resume.strip() or not job.strip():
        return 0.0

    embeddings = model.encode(
        [resume, job],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    similarity = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0]

    return float(max(0.0, min(1.0, similarity)) * 100)