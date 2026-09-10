import re

DEGREE_EQUIVALENTS = {
    "bachelor": [
        "bachelor",
        "bachelors",
        "bachelor's",
        "b.e",
        "be",
        "b.tech",
        "btech",
        "b.sc",
        "bsc",
        "bca"
    ],
    "master": [
        "master",
        "master's",
        "masters",
        "m.e",
        "me",
        "m.tech",
        "mtech",
        "m.sc",
        "msc",
        "mca"
    ],
    "phd": [
        "phd",
        "doctorate"
    ]
}


def normalize(text):
    text = text.lower()

    text = re.sub(r'[^a-z0-9\s.]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text


def education_score(resume, job):

    resume = normalize(resume)
    job = normalize(job)

    # Find which degree is required
    required_degree = None

    for degree, aliases in DEGREE_EQUIVALENTS.items():
        for alias in aliases:
            if alias in job:
                required_degree = degree
                break
        if required_degree:
            break

    # No education mentioned in JD
    if required_degree is None:
        return 100

    # Check if resume has any equivalent degree
    for alias in DEGREE_EQUIVALENTS[required_degree]:
        if alias in resume:
            return 100

    return 0