#Star vs galaxy
from sklearn.metrics import accuracy_score
from astroML.datasets import fetch_sdss_specgals
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = fetch_sdss_specgals()
df = pd.DataFrame(data)

X = df[['modelMag_u','modelMag_g','modelMag_r','d4000']]
y = df['bptclass']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test  =scaler.transform(X_test)

#model = RandomForestClassifier(n_estimators=100)
model = LogisticRegression()
model.fit(X_train, y_train)
unseen_data = pd.DataFrame([
    [18.2, 17.1, 16.5, 1.8],
    [20.1, 19.3, 18.7, 1.2],
    [17.5, 16.8, 16.0, 2.1],
    [21.0, 20.2, 19.5, 1.0],
    [19.0, 18.2, 17.6, 1.6]
], columns=['modelMag_u', 'modelMag_g', 'modelMag_r', 'd4000'])
unseen_data_scaled = scaler.transform(unseen_data)
y_pred = model.predict(unseen_data_scaled)
y_test_pred = model.predict(X_test)
print(accuracy_score(y_test, y_test_pred))
print(y_pred)