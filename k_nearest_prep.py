import acquire
import prep
import pandas as pd
import numpy as np
from env import seed
from sklearn.preprocessing import MinMaxScaler

telco = acquire.get_telco()

telco = prep.prep_telco(telco, False)

telco.drop(columns=['family', 'tenure_years'], inplace=True)
# MAKE EVERY VARIABLE BETWEEN ONE AND ZERO
to_drop = ['pay', 'internet', 'contract', 'phone']
to_map = ['security', 'backup', 'protection', 'support', 'tv', 'movies']

telco['e_check'] = (telco['pay'] == 1).astype(int)
telco['mail_check'] = (telco['pay'] == 2).astype(int)
telco['bank'] = (telco['pay'] == 3).astype(int)
telco['cc'] = (telco['pay'] == 4).astype(int)

telco['no_internet'] = (telco['internet'] == 0).astype(int)
telco['fiber'] = (telco['internet'] == 1).astype(int)
telco['dsl'] = (telco['internet'] == 2).astype(int)

telco['no_phone'] = (telco['phone'] == 0).astype(int)
telco['one_line'] = (telco['phone'] == 1).astype(int)
telco['mult_lines'] = (telco['phone'] == 2).astype(int)

telco['biyearly'] = (telco['contract'] == 3).astype(int)
telco['yearly'] = (telco['contract'] == 1).astype(int)
telco['month_to_month'] = (telco['contract'] == 2).astype(int)

for feature in to_map:
    telco[feature] = telco[feature].map({0:0, 1:0, 2:1})

telco.drop(columns=(to_drop + to_map), inplace=True)

# scale tenure, monthly, total
to_scale = ['tenure', 'monthly', 'total']

for feature in to_scale:
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(telco[[feature]])
    telco[feature] = scaler.transform(telco[[feature]])