from env import user, host, password
import pandas as pd

# given identifying information, returns sql database url
def get_db_url(username, hostname, password, db_name):
    return f'mysql+pymysql://{username}:{password}@{hostname}/{db_name}'

# returns a dataframe of the telco_churn database
def get_telco():
    db_name = 'telco_churn'

    url = get_db_url(user, host, password, db_name)

    # Gets all the data from sql, including labels for each type ID
    query = '''select * from customers
                join contract_types using(contract_type_id)
                join internet_service_types using(internet_service_type_id)
                join payment_types using(payment_type_id);
                '''

    return pd.read_sql(query, url)

