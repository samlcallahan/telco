import acquire
import prep
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from features import optimal_feature_n, top_n_features
import warnings
from env import seed
warnings.filterwarnings("ignore")

telco = acquire.get_telco()
X_train, X_test, y_train, y_test, data_dict = prep.prep_telco(telco)

first_model = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3, max_depth=8, random_state=seed)
first_model.fit(X_train, y_train)
y_pred = first_model.predict(X_train)
accuracy = first_model.score(X_train, y_train) # 80% on first try, 55% recall is pretty awful though
recall = classification_report(y_train, y_pred, output_dict=True)['1']['recall']

top_n_features(X_train, y_train, optimal_feature_n(X_train, y_train, first_model), first_model)
