import acquire
from env import seed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

telco = acquire.get_telco()

# used to find odd values that needed correction
# for i in telco.columns:
#     print(telco[i].value_counts(dropna=False))

def can_it_float(value):
    ''' a function to check whether a value will successfully cast to float
        returns True if can be cast to float, otherwise False
    '''
    try:
        float(value)
    except:
        return False
    else:
        return True

def yes_no_encoder(telco_column):
    return telco_column.map({'Yes': 1, 'No': 0})

def yes_no_none(value):
    if value == "Yes":
        return 2
    elif value == "No":
        return 1
    else:
        return 0

def yes_no_none_encoder(telco_column):
    return telco_column.map(yes_no_none)


def prep_telco(telco_df):
    # Finds all total_charges which cannot be cast to float
    invalid_totals = ~telco.total_charges.apply(can_it_float)

    # After investigating those, I found they all had a tenure of 0, so I figured making total_charges = 0 made sense
    telco[invalid_totals] = 0

    telco['total_charges'] = telco['total_charges'].astype('float')

    telco['tenure_years'] = telco['tenure'] / 12

    data_dict = {}
    payment = {1: 'electronic check',
                2: 'mailed check',
                3: 'bank',
                4: 'credit card'}
    data_dict['payment'] = payment

    contract = {1: 'monthly',
                2: 'yearly',
                3: 'biyearly'}
    data_dict['contract'] = contract

    internet = {1: 'dsl',
                2: 'fiber',
                3: 'none'}
    data_dict['internet'] = internet

    # drop these since the same data is in the respective ID columns
    telco.drop(columns=['payment_type', 'internet_service_type', 'contract_type'], inplace=True)

    # make customer_id the index so it isn't considered as a feature
    telco.set_index('customer_id', inplace=True)

    # rename some things to accomodate my laziness
    rename_dict = {'payment_type_id' : 'pay',
                    'internet_service_type_id' : 'internet',
                    'contract_type_id' : 'contract',
                    'senior_citizen' : 'senior',
                    'phone_service' : 'phone',
                    'multiple_lines' : 'lines',
                    'online_security' : 'security',
                    'online_backup' : 'backup',
                    'device_protection' : 'protection',
                    'tech_support' : 'support',
                    'streaming_tv' : 'tv',
                    'streaming_movies' : 'movies',
                    'paperless_billing' : 'paperless',
                    'monthly_charges' : 'monthly',
                    'total_charges' : 'total'}
    telco.rename(columns=rename_dict, inplace=True)

    yes_no_cols = ['churn', 'partner', 'dependents', 'paperless', 'phone']
    for i in yes_no_cols:
        telco[i] = yes_no_encoder(telco[i])

    gender = {'Male' : 1, 'Female' : 0}
    data_dict['gender'] = gender
    telco.gender = telco.gender.map(gender)

    yes_no_other_cols = ['lines', 'security', 'backup', 'protection', 'support', 'tv', 'movies']
    for i in yes_no_other_cols:
        telco[i] = yes_no_none_encoder(telco[i])

    y = telco[['churn']]
    X = telco.drop(columns='churn')
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed)
    return X_train, x_test, y_train, y_test, data_dict