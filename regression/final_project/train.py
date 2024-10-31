import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

# custom files
import model_best_hyperparameters
import columns

# read train data
df = pd.read_csv("train_data.csv")

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

# Save parameters
param_dict = {'upper_lower_limits': upper_lower_limits}
with open('param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Define target and features columns
X = df[columns.X_columns]
y = df[columns.y_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

# Initialize XGBRegressor with best hyperparameters
xgb = xgb.XGBRegressor(**model_best_hyperparameters.params)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# Print test set metrics
test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
test_mae = metrics.mean_absolute_error(y_test, y_pred)
test_r2 = metrics.r2_score(y_test, y_pred)

print("Test Set Metrics:")
print(f'RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R2: {test_r2:.2f}')

# Save the final model
filename = 'finalized_model.sav'
pickle.dump(xgb, open(filename, 'wb'))
