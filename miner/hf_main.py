import mysql.connector
from mysql.connector import Error
import config as cf
import data_utils as du

def create_server_connection(host_name, user_name, user_password, db_name, port_number):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name,
            port=port_number
        )
        print("MySQL Database connection successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection



def get_dataset_description(conn):
    dict_datasets = {}
    cur = conn.cursor()
    cur.execute("SELECT id,description FROM dataset,repository where dataset.dataset_id=repository.id;")
    data = cur.fetchall()
    for d in data:
        dict_datasets.update({d[0]: d[1]})
    return dict_datasets



if __name__ == '__main__':
    connection = create_server_connection(cf.HOST, cf.USER, cf.PWD, cf.DB, cf.PORT)
    model_dict = get_dataset_description(connection)
    du.write_dict_to_csv(model_dict,'hf_dump.csv','dataset_id', 'desc')
    if connection.is_connected():
        connection.close()
        print("MySQL connection is closed")
