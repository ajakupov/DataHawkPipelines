import argparse
import os
import pandas as pd
import numpy as np
import math
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import mean_squared_error

print("In train.py")
print("As a data scientist, this is where I write my training code.")

parser = argparse.ArgumentParser("train")

parser.add_argument("--input", type=str, help="input directory", dest="input", required=True)
parser.add_argument("--output", type=str, help="output directory", dest="output", required=True)

args = parser.parse_args()

print("Argument 1: %s" % args.input)
print("Argument 2: %s" % args.output)

# Load your processed features and outputs
df = pd.read_csv(os.path.join(args.input, 'nyc-taxi-processed-data.csv'))

x_df = df.drop(['totalAmount'], axis=1)
y_df = df['totalAmount']

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=0)

# we will not transform / scale the four engineered features:
# hour_sine, hour_cosine, day_of_week_sine, day_of_week_cosine
categorical = ['normalizeHolidayName', 'isPaidTimeOff']
numerical = ['vendorID', 'passengerCount', 'tripDistance', 'day_of_month', 'month_num', 
             'snowDepth', 'precipTime', 'precipDepth', 'temperature']

numeric_transformations = [([f], Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])) for f in numerical]
    
categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]

transformations = numeric_transformations + categorical_transformations

# df_out will return a data frame, and default = None will pass the engineered features unchanged
mapper = DataFrameMapper(transformations, input_df=True, df_out=True, default=None, sparse=False)

clf = Pipeline(steps=[('preprocessor', mapper),
                      ('regressor', GradientBoostingRegressor())])

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
y_actual = y_test.values.flatten().tolist()
rmse = math.sqrt(mean_squared_error(y_actual, y_predict))
print('The RMSE score on test data for GradientBoostingRegressor: ', rmse)

# Save the model
if not (args.output is None):
    os.makedirs(args.output, exist_ok=True)
    output_filename = os.path.join(args.output, 'nyc-taxi-fare.pkl')
    pickle.dump(clf, open(output_filename, 'wb'))
    print('Model file nyc-taxi-fare.pkl saved!')
