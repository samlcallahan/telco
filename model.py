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
from k_nearest_prep import knn_df
from env import seed
warnings.filterwarnings("ignore")

telco = acquire.get_telco()

X_train, X_test, y_train, y_test, data_dict = prep.prep_telco(telco)

first_model = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3, max_depth=8, random_state=seed)
first_model.fit(X_train, y_train)
y_pred = first_model.predict(X_train)
accuracy = first_model.score(X_train, y_train) # 80% on first try, 55% recall isn't very good though
recall = classification_report(y_train, y_pred, output_dict=True)['1']['recall']

rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=seed)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_train)
accuracy = rf_model.score(X_train, y_train)

X = knn_df.drop(columns='churn')
y = knn_df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed)
knn = KNeighborsClassifier(weights='distance', n_neighbors = 15)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_train)
recall = classification_report(y_train, y_pred, output_dict=True)['1']['recall']
accuracy = knn.score(X_train, y_train)


# FIRST MODEL LOOKS THE BEST.