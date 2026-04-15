#Beginner project of(load_breast_cancer)
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)
X,y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 42)

#Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Training the model

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)
new_patient = np.array([[
    14.2, 20.1, 92.5, 600.1, 0.1, 0.15, 0.2, 0.1, 0.18, 0.06,
    0.3, 0.5, 1.2, 20.0, 0.006, 0.02, 0.03, 0.01, 0.02, 0.004,
    16.0, 25.0, 105.0, 700.0, 0.13, 0.25, 0.3, 0.12, 0.25, 0.08
]])
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)
y_pred = model.predict(X_test)
#####print(accuracy_score(y_test,y_pred))

#
if prediction[0] == 1:
    print("Benign, Patient has no cancer")
else:
    print("Malignant, patient has cancer")
print("The accuracy of the model is", accuracy_score(y_test,y_pred))