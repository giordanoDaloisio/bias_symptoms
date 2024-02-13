def get_label(dataset):
    if "adult" in dataset:
        return ("income", 1)
    # if "aps" in dataset:
    #     return "class"
    if "cmc" in dataset:
        return ("contr_use", 2)
    if "compas" in dataset:
        return ("two_year_recid", 0)
    if "crime" in dataset:
        return ("ViolentCrimesClass", 100)
    if "drug" in dataset:
        return ("y", 0)
    if "german" in dataset:
        return ("credit", 1)
    if "healt" in dataset:
        return ("y", 1)
    if "hearth" in dataset:
        return ("y", 0)
    # if "kickstarter" in dataset:
    #     return "State"
    if "law" in dataset:
        return ("gpa", 2)
    if "medical" in dataset:
        return ("IsChallenge", 0)
    if "obesity" in dataset:
        return ("y", 0)
    if "park" in dataset:
        return ("score_cut", 0)
    if "pop_bias" in dataset:
        return ("ranking", 1)
    if "resyduo" in dataset:
        return ("tot_recommendations", 1)
    if "student" in dataset:
        return ("y", 1)
    if "wine" in dataset:
        return ("quality", 6)
    return ("y", 1)
