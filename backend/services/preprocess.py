import re


def clean_text(text):
    text = text.lower()

    # Fix 1: preserve tech symbols (+ # . - /) so skills like c++, c# and
    # scikit-learn survive cleaning and stay distinguishable downstream.
    text = re.sub(r"[^a-z0-9 +#.\-/]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()