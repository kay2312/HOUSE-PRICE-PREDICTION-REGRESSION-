import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

# custom files
import columns

# read data
df = pd.read_csv("new_data.csv")
print('new data size', df.shape)

# feature engineering
param_dict = pickle.load(open('param_dict.pickle', 'rb'))

# feature engineering
# Outlier Engineering with Capping
def find_skewed_boundaries(df, variable, distance):
    df[variable] = pd.to_numeric(df[variable], errors='coerce')
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary

upper_lower_limits = dict()
for column in columns.outlier_columns:
    upper_lower_limits[column + '_upper_limit'], upper_lower_limits[column + '_lower_limit'] = find_skewed_boundaries(df, column, 3)

# Replacing outliers with upper or lower boundary
for column in columns.outlier_columns:
    upper_limit = upper_lower_limits[column + '_upper_limit']
    lower_limit = upper_lower_limits[column + '_lower_limit']
    
    # Apply capping for values above upper limit and below lower limit
    df[column] = np.where(df[column] > upper_limit, upper_limit,
                          np.where(df[column] < lower_limit, lower_limit, df[column]))

# Define features columns
X = df[columns.X_columns]

# load the model and predict
xgb = pickle.load(open('finalized_model.sav', 'rb'))

y_pred = xgb.predict(X)

df['Price_pred'] = xgb.predict(X)
df.to_csv('prediction_results.csv', index=False)