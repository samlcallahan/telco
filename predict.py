from model import first_model
import acquire
import pandas as pd
import prep

telco = acquire.get_telco()
X_train, X_test, y_train, y_test, data_dict = prep.prep_telco(telco)

X = pd.concat([X_train, X_test])

results = pd.DataFrame(index=X.index)

results['prediction'] = first_model.predict(X)

results['probability'] = first_model.predict_proba(X)[:,1]

results.to_csv('results.csv')