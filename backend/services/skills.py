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


def _pattern(skill):
    # Fix 1: \b fails after symbol chars ("c++" never matches \bc\+\+\b),
    # so symbol skills use lookarounds; plain skills keep \b.
    if re.fullmatch(r"[a-z0-9][a-z0-9 .\-]*", skill):
        return r"\b" + re.escape(skill) + r"\b"
    return r"(?<!\w)" + re.escape(skill) + r"(?!\w)"


# Fix 3: standalone "c" matches grades ("grade C"), "vitamin C", "option C".
# Require programming context or a companion C-family skill.
_C_CONTEXT = [
    "programming", "language", "software", "developer", "code", "coding",
    "embedded", "firmware", "compiler", "algorithm", "computer science",
    "debugging", "microcontroller",
]


def _c_context_ok(text):
    if re.search(r"(?<!\w)c\+\+(?!\w)", text) or re.search(r"(?<!\w)c#(?!\w)", text):
        return True
    return any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in _C_CONTEXT)


def extract_skills(text):
    text = text.lower()

    found = set()

    for skill in SKILLS:
        if skill == "c":
            continue  # handled with context below

        if re.search(_pattern(skill), text):
            found.add(skill)

    if re.search(r"(?<!\w)c(?!\w)", text) and _c_context_ok(text):
        found.add("c")

    return found