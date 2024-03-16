import os
import time

# import pyRAPL
import pandas as pd
from joblib import load
from symp_extractor import symp_extractor


def get_label_var(dataset):
    if "adult" in dataset:
        return ("income", 1, "sex")
    if "arrhythmia" in dataset:
        return ("y", 1, 1)
    if "bank" in dataset:
        return ("loan", 1, "age")
    if "cmc" in dataset:
        return ("contr_use", 2, "wife_religion")
    if "compas" in dataset:
        return ("two_year_recid", 0, "race")
    if "crime" in dataset:
        return ("ViolentCrimesClass", 100, "black_people")
    if "credit_card" in dataset:
        return ("y", 1, "SEX")
    if "diabetic" in dataset:
        return ("diabetesMed", 0, "gender_Female")
    if "drug" in dataset:
        return ("y", 0, "gender")
    if "german" in dataset:
        return ("credit", 1, "age")
    if "healt" in dataset:
        return ("y", 1, "sexFEMALE")
    if "hearth" in dataset:
        return ("y", 0, "sex")
    if "ibm" in dataset:
        return ("Attrition", 0, "Gender_Female")
    if "law" in dataset:
        return ("gpa", 2, "race")
    if "medical" in dataset:
        return ("IsChallenge", 0, "gender")
    if "obesity" in dataset:
        return ("y", 0, "Gender")
    if "park" in dataset:
        return ("score_cut", 0, "sex")
    if "placement" in dataset:
        return ("status", 1, "gender_F")
    if "resyduo" in dataset:
        return ("tot_recommendations", 1, "views")
    if "ricci" in dataset:
        return ("Combine", 1, "Race_B")
    if "student" in dataset:
        return ("y", 1, "sex_M")
    if "us" in dataset:
        return ("dIncome1", 3, "iSex")
    if "vaccine" in dataset:
        return ("lowtrustvaccinerec", 0, "female")
    if "wine" in dataset:
        return ("quality", 6, "type")
    return ("y", 1)


if __name__ == "__main__":
    model = load("model.joblib")
    # pyRAPL.setup()
    # measure = pyRAPL.Measurement("bar")
    # csv_output = pyRAPL.outputs.CSVOutput("measures.csv")
    times = []
    os.makedirs("our_approach_results", exist_ok=True)
    for i in range(20):
        for file in os.listdir("../data"):
            data = pd.read_csv(f"../data/{file}")
            label, pos_label, sensitive_var = get_label_var(file)
            # measure.begin()
            start_time = time.time()
            symptoms = symp_extractor(data, label, pos_label, sensitive_var)
            pred = model.predict(symptoms)
            end_time = time.time()
            # measure.end()
            # measure.export(csv_output)
            times.append(end_time - start_time)
            print(f"Dataset: {file} completed")
            pd.DataFrame(pred).to_csv(f"our_approach_results/{file}_report.csv")
        print(f"Round: {i} completed")
    with open("times.txt", "w") as f:
        for time in times:
            f.write(str(time) + "\n")
    # csv_output.save()
