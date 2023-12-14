import time

import requests
from bs4 import BeautifulSoup
from ucimlrepo import fetch_ucirepo, list_available_datasets
from miner import data_utils as du
import pandas as pd
import config as cf


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

def collect_uci_datasets(out_file):
    list_tuples = du.generate_range_ratings(0, 100, 25)

    dict_links = {}
    index = 0
    for start, end in list_tuples:
        list_url = 'https://archive.ics.uci.edu/datasets?Task=Classification&skip=+'+str(start)+'&take='+str(end)+'&sort=desc&orderBy=NumHits&search=&Types=Tabular'
        list_links = parse_html_to_json(list_url)
        time.sleep(5)
        for link in list_links:
            id_dataset=du.extract_numbers(link)
            dict_links.update({id_dataset: link})
            print('updating dict')

    du.write_dict_to_csv(dict_links, out_file)



def collect_uci_metadata(file_name, dump):
    collected_dataset = list_available_datasets()
    df_dump = pd.read_csv(dump)
    #list_dump = df_dump['id'].values.astype(int)
    du.create_folder_if_not_exist(cf.UCI_METADATA_FILE)
    #list_datasets = set(collected_dataset + list_dump)
    out_file = cf.UCI_METADATA_FILE + file_name
    pd.DataFrame(columns=cf.UCI_COLUMNS).to_csv(out_file, index=False)
    for dict_data in collected_dataset:
        data = {}
        try:
            df_uci = fetch_ucirepo(id=dict_data.get('id'))

            data.update({"id": df_uci.metadata.uci_id ,
                         "url": df_uci.metadata.data_url ,
                         "task": df_uci.metadata.task,
                         "summary": str(df_uci.metadata.additional_info.summary).strip(),
                         "sensitive_data": str(df_uci.metadata.additional_info.sensitive_data).strip(),
                         "variable_info": str(df_uci.metadata.additional_info.variable_info).strip() })
            df_meta = pd.DataFrame([data])
            df_meta.to_csv(out_file, mode='a', header=False, index=False)
            print('mining uci metadata')
        except:
            continue


    # for id_dataset, url in zip(df_meta['id'].values.astype(str),df_meta['url'].values.astype(str)):
    #     download_dataset(url, id_dataset+'.csv')



if __name__ == "__main__":
    df_data = pd.read_csv('uci_data/result.csv')
    # for id, url in zip(df_data['id'].values.astype(str),df_data['url'].values.astype(str)):
    #     download_dataset(url, id+'.csv')
    #collect_uci_metadata('dump_data.csv', cf.UCI_DUMP_FILE)
    du.compute_variable_stats('uci_data/datasets/', 'vars_stats.csv')
