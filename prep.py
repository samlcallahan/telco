import acquire
import pandas as pd
import numpy as np

def df_value_counts(dataframe):
    for col in dataframe.columns:
        if dataframe[col].nunique() == len(dataframe[col]):
            continue
        elif dataframe[col].nunique() > 10:
            print(dataframe[col].value_counts(bins=10, dropna=False))
        else:
            print(dataframe[col].value_counts(dropna=False))

telco = acquire.get_telco()

def can_it_float(value):
    try:
        float(value)
    except:
        return False
    else:
        return True

invalid_totals = ~telco.total_charges.apply(can_it_float)

telco['total_charges'] = telco['total_charges'].astype('float')
