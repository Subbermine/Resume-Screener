import re

SKILLS = {
    "python","java","c","c++","c#","javascript",
    "typescript","html","css","react","angular","vue",
    "node","express","mongodb","mysql",
    "postgresql","oracle","sql","docker","kubernetes",
    "aws","azure","gcp","git","github","linux",
    "spring","spring boot",
    "hibernate","fastapi",
    "flask","django",
    "tensorflow","pytorch",
    "machine learning","deep learning",
    "opencv","pandas",
    "numpy","scikit-learn"
}


def extract_skills(text):
    text = text.lower()

    found = set()

    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):
            found.add(skill)

    return found