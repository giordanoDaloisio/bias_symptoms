import requests
from bs4 import BeautifulSoup
import json
from ucimlrepo import fetch_ucirepo, list_available_datasets
import os
import pandas as pd
import config as cf

def create_folder_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def parse_html_to_json(url, json_file_name):
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Create a dictionary to store the parsed data
        parsed_data = {
            'title': soup.title.string if soup.title else None,
            'body': soup.get_text(),
        }

        # Save the parsed data as JSON
        with open(json_file_name, 'w', encoding='utf-8') as json_file:
            json.dump(parsed_data, json_file, indent=4, ensure_ascii=False)
        print(f'HTML content from {url} has been successfully parsed and saved as {json_file_name}.')

        return parsed_data

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
    df_data = pd.read_csv('uci_metadata/result.csv')
    print(df_data.shape)

# # Example usage:
# url = 'https://archive.ics.uci.edu/datasets?Task=Classification&skip=0&take=10&sort=desc&orderBy=NumHits&search=&Types=Tabular'  # Replace with the URL you want to parse
# json_file_name = 'parsed_data.json'  # Replace with the desired JSON file name
# parse_html_to_json(url, json_file_name)
