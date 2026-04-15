#stock price prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import mean_squared_error
df = pd.read_csv("google_stock_data.csv")
df["MA5"] = df["Close"].rolling(5).mean()
df["MA10"] = df["Close"].rolling(10).mean()
df["Momentum"]= df["Close"] - df["Close"].shift(1)
df = df.dropna()
df["direction"]= (df["Close"].shift(-1)> df["Close"]).astype(int)
df["price_target"] = df["Close"].shift(-1)
df = df.dropna()
X = df[["Open", "High", "Low", "Close", "Volume","MA5","MA10","Momentum"]]
y_dir = df["direction"]
y_price = df["price_target"]
X_train, X_test, y_dir_train, y_dir_test, y_price_train, y_price_test= (
    train_test_split(X, y_dir,
                     y_price,
    shuffle = False,
    test_size = 0.2))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
price_model = LinearRegression()
price_model.fit(X_train, y_price_train)
price_pred = price_model.predict(X_test)
dir_model = RandomForestClassifier()
dir_model.fit(X_train, y_dir_train)
dir_pred = dir_model.predict(X_test)
if dir_pred[0] == 1:
    print("BUY ! price may go up")
else:
    print("SELL ! price may go down")
print(price_pred[0])
print("Direction accuracy:",accuracy_score(y_dir_test, dir_pred))
print("price MSE:", mean_squared_error(y_price_test, price_pred))