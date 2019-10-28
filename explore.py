import acquire
import prep
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

telco = acquire.get_telco()
X_train, X_test, y_train, y_test, data_dict = prep.prep_telco(telco)

sns.heatmap(X_train.corr(), cmap='Blues')

