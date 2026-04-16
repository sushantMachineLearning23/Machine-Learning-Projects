#Imports

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Load Data
df = pd.read_csv("AirQuality.csv", sep=';')

#Cleaning the noisy Data
df.drop(["Date", "Time"], axis=1, inplace=True)
df.replace([-200, -999], np.nan, inplace=True)
df = df.replace(',', '.', regex=True)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.mean())
df = df[df['CO(GT)']>0]
#Split
X = df.drop('CO(GT)', axis=1)
y = df['CO(GT)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#model selection
model = RandomForestRegressor(n_estimators=100, random_state=42)#
model.fit(X_train, y_train)
new_data = pd.DataFrame()

#Evaluation
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

#user input
print("Air Quality Model")
input_data = []
for col in X.columns:
    val = float(input(f"Enter value {col}:"))
    input_data.append(val)
input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)
prediction = model.predict(input_scaled)[0]
print(prediction)
new_prediction = model.predict(X_test)
print(new_prediction)
threshold = df["CO(GT)"].median()
print(threshold)
if prediction > threshold:
    print("Unsafe to visit")
else:
    print("Safe to visit")