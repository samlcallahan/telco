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
import warnings
from env import seed
warnings.filterwarnings("ignore")

telco = acquire.get_telco()
X_train, X_test, y_train, y_test, data_dict = prep.prep_telco(telco)

first_model = DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, max_depth=10, random_state=seed)
first_model.fit(X_train, y_train)
y_pred = first_model.predict(X_train)
score = first_model.score(X_train, y_train) # 84% on first try! that might be hard to beat