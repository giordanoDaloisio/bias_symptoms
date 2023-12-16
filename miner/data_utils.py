import csv
import pandas as pd
import config as cf
import os
import re



def filter_hf_datasets(hf_path):
    hf_dump = pd.read_csv(hf_path)
    df_res = pd.DataFrame(columns=hf_dump.columns)
    keyword_set = ['bias', 'fairness', 'discrimination']
    for idx, row in hf_dump.iterrows():
        for k in keyword_set:
            if k in str(row):
                df_res = pd.concat([df_res, pd.DataFrame([row])], ignore_index=True)

    df_res.to_csv('hf_dump_filtered.csv', index=False)





def preprocess_variable_name(var_name):
    return str(var_name).lower()

def write_dict_to_csv(dict, out_file, field_key, field_val):
    with open(out_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [field_key, field_val]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in dict.items():
            if key is not None:  # Ensure that key is not None
                writer.writerow({field_key: key, field_val : value})


def compute_variable_stats(src_folder, out_file):
    list_variables = []
    for data in os.listdir(src_folder):
        df_data = pd.read_csv(src_folder + data)
        for var in df_data.columns:
            list_variables.append(preprocess_variable_name(var))
    count_occurrences(list_variables, out_file)


def extract_numbers(s):
    match = re.search(r'\d+', s)
    return match.group(0) if match else None


def count_occurrences(elements, filename):

    frequency = {}
    for element in elements:
        frequency[element] = frequency.get(element, 0) + 1

    # Sorting the dictionary by frequency
    sorted_frequency = dict(sorted(frequency.items(), key=lambda item: item[1], reverse=True))

    # Writing to CSV file
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Element', 'Frequency'])  # Writing headers
        for key, value in sorted_frequency.items():
            writer.writerow([key, value])

    print(f"Data saved to {filename}")


def append_and_remove_duplicates(df1, df2, subset=None):
    combined_df = pd.concat([df1, df2])
    unique_df = combined_df.drop_duplicates(subset=subset)
    return unique_df


def generate_range_ratings(start, end, delta):
    tuple_list = []
    current = start
    while current <= end - delta:
        tuple_list.append((current, current + delta))
        current += delta
    return tuple_list


def create_folder_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
