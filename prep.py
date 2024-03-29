import acquire
from env import seed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

telco = acquire.get_telco()

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

# Encodes a column where "Yes" = 1 and "No" = 0
def yes_no_encoder(telco_column):
    return telco_column.map({'Yes': 1, 'No': 0})

def yes_no_none(value):
    if value == "Yes":
        return 2
    elif value == "No":
        return 1
    else:
        return 0

# Encodes a column where "Yes" = 2, "No" = 1, and anything else is 0
def yes_no_none_encoder(telco_column):
    return telco_column.map(yes_no_none)

# if a value can be cast to float, returns that value, otherwise returns 0
def cant_float_to_zero(value):
    if can_it_float(value):
        return value
    else:
        return 0

def prep_telco(telco_df, split=True):
    # Looked up non-floatable values in total_charges
    # After investigating those, I found they all had a tenure of 0, so I figured making total_charges = 0 made sense
    prepped = telco_df
    
    # Sets all abnormal total_charges to 0
    prepped.total_charges = prepped.total_charges.map(cant_float_to_zero)

    # Casts total_charges to float
    prepped['total_charges'] = prepped['total_charges'].astype('float')

    prepped['tenure_years'] = prepped['tenure'] / 12

    # Dictionary in which my data encodings are explained
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

    internet = {0: 'none',
                1: 'dsl',
                2: 'fiber'}
    data_dict['internet'] = internet

    # drop these since the same data is in the respective ID columns
    prepped.drop(columns=['payment_type', 'internet_service_type', 'contract_type'], inplace=True)

    # make customer_id the index so it isn't considered as a feature
    prepped.set_index('customer_id', inplace=True)

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
    prepped.rename(columns=rename_dict, inplace=True)

    # Maps internet to match data dictionary and be more sensible.
    # Converts No Internet from 3 to 0, which makes more sense since it's "less" internet
    prepped.internet = prepped.internet.map({1:1, 2:2, 3:0})

    # Takes columns whose entries are "yes" or "no" and maps them to 1 or 0, respectively
    yes_no_cols = ['churn', 'partner', 'dependents', 'paperless', 'phone']
    for i in yes_no_cols:
        prepped[i] = yes_no_encoder(prepped[i])

    # Maps male gender to 1, female to 0
    gender = {'Male' : 1, 'Female' : 0}
    data_dict['gender'] = gender
    prepped.gender = prepped.gender.map(gender)

    # Encodes columns whose options boil down to "yes," "no," or "NA" to be 2, 1, or 0 respectively
    yes_no_other_cols = ['lines', 'security', 'backup', 'protection', 'support', 'tv', 'movies']
    for i in yes_no_other_cols:
        prepped[i] = yes_no_none_encoder(prepped[i])

    # Combines phone and lines feature, and drops original lines feature.
    prepped.loc[prepped.lines == 2, 'phone'] = 2
    prepped.drop(columns='lines', inplace=True)
    data_dict['phone'] = {0: 'no phone', 1: 'one line', 2: 'multiple lines'}

    # Combines partner and dependents into family. Encoding listed in data dictionary.
    prepped['family'] = prepped.partner + (2 * prepped.dependents)
    data_dict['family'] = {0: 'no partner, no dependents',
                        1: 'partner, no dependents',
                        2: 'no partner, dependents',
                        3: 'partner and dependents'}
    prepped.drop(columns=['dependents', 'partner'])

    y = prepped[['churn']]
    X = prepped.drop(columns='churn')
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed)

    # Returns split data unless we are going to use this funciton in our k-nearest neighbors prep, which was easier with unsplit data.
    if split:
        return X_train, x_test, y_train, y_test, data_dict
    else:
        return prepped