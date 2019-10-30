import acquire
import prep
import pandas as pd
import numpy as np
from env import seed
from sklearn.preprocessing import MinMaxScaler

knn_df = acquire.get_telco()

knn_df = prep.prep_telco(knn_df, False)

knn_df.drop(columns=['family', 'tenure_years'], inplace=True)
# MAKE EVERY VARIABLE BETWEEN ONE AND ZERO
to_drop = ['pay', 'internet', 'contract', 'phone']
to_map = ['security', 'backup', 'protection', 'support', 'tv', 'movies']

knn_df['e_check'] = (knn_df['pay'] == 1).astype(int)
knn_df['mail_check'] = (knn_df['pay'] == 2).astype(int)
knn_df['bank'] = (knn_df['pay'] == 3).astype(int)
knn_df['cc'] = (knn_df['pay'] == 4).astype(int)

knn_df['no_internet'] = (knn_df['internet'] == 0).astype(int)
knn_df['fiber'] = (knn_df['internet'] == 1).astype(int)
knn_df['dsl'] = (knn_df['internet'] == 2).astype(int)

knn_df['no_phone'] = (knn_df['phone'] == 0).astype(int)
knn_df['one_line'] = (knn_df['phone'] == 1).astype(int)
knn_df['mult_lines'] = (knn_df['phone'] == 2).astype(int)

knn_df['biyearly'] = (knn_df['contract'] == 3).astype(int)
knn_df['yearly'] = (knn_df['contract'] == 1).astype(int)
knn_df['month_to_month'] = (knn_df['contract'] == 2).astype(int)

for feature in to_map:
    knn_df[feature] = knn_df[feature].map({0:0, 1:0, 2:1})

knn_df.drop(columns=(to_drop + to_map), inplace=True)

# scale tenure, monthly, total
to_scale = ['tenure', 'monthly', 'total']

for feature in to_scale:
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(knn_df[[feature]])
    knn_df[feature] = scaler.transform(knn_df[[feature]])