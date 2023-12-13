import time

import requests
from bs4 import BeautifulSoup
import json
from ucimlrepo import fetch_ucirepo, list_available_datasets
import os
import pandas as pd
import config as cf
import re
import csv

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

def download_dataset(url, save_path):
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Open the file in binary write mode and write the content to it
        with open(cf.UCI_METADATA_FILE + save_path, 'wb') as file:
            file.write(response.content)

        print(f'File downloaded from {url} and saved as {save_path}.')
    except Exception as e:
        print(f'An error occurred: {str(e)}')


def extract_numbers(s):
    match = re.search(r'\d+', s)
    return match.group(0) if match else None


def parse_html_to_json(url_repo):
    list_links = []
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url_repo)
        response.raise_for_status()

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        for link in soup.find_all('a'):
            if 'dataset/' in str(link):
                list_links.append(link.get('href'))

        return set(list_links)

    except Exception as e:
        print(f'An error occurred: {str(e)}')


def collect_uci_metadata():
    list_datasets = list_available_datasets()
    create_folder_if_not_exist(cf.UCI_METADATA_FILE)

    pd.DataFrame(columns=cf.UCI_COLUMNS).to_csv(cf.UCI_METADATA_FILE + 'result.csv', index=False)

    #['id', 'url', 'task' 'summary', 'sensitive_data', 'variable_info']
    for dat in list_datasets:
        data = {}
        df_uci = fetch_ucirepo(id=dat.get('id'))



        data.update({"id": df_uci.metadata.uci_id ,
                     "url" : df_uci.metadata.data_url ,
                     "task": df_uci.metadata.task,
                     "summary": str(df_uci.metadata.additional_info.summary).strip(),
                     "sensitive_data": str(df_uci.metadata.additional_info.sensitive_data).strip(),
                     "variable_info": str(df_uci.metadata.additional_info.variable_info).strip() })
        df_meta = pd.DataFrame([data])
        df_meta.to_csv(cf.UCI_METADATA_FILE + 'result.csv', mode='a', header=False, index=False)




if __name__ == "__main__":
    #df_data = pd.read_csv('uci_data/result.csv')
    # for id, url in zip(df_data['id'].values.astype(str),df_data['url'].values.astype(str)):
    #     download_dataset(url, id+'.csv')
    list_tuples = generate_range_ratings(0, 100, 25)

    dict_links = {}
    index = 0
    for start, end in list_tuples:
        list_url = 'https://archive.ics.uci.edu/datasets?Task=Classification&skip=+'+str(start)+'&take='+str(end)+'&sort=desc&orderBy=NumHits&search=&Types=Tabular'
        list_links = parse_html_to_json(list_url)
        time.sleep(5)
        for link in list_links:
            id_dataset=extract_numbers(link)
            dict_links.update({id_dataset: link})
            print('updating dict')

    with open('uci_data/links.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'link']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key, value in dict_links.items():
            if key is not None:  # Ensure that key is not None
                writer.writerow({'id': key, 'link': value})