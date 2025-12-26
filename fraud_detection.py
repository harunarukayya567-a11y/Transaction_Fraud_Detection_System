import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib


data = pd.read_csv("fraud_dataset2.csv")
df = pd.DataFrame(data)

print(df.head(9))

df = df.drop(['UserID', 'TransactionID'], axis=1)

y = df['IsFraud']
X = df.drop(['IsFraud'], axis=1)

ct = ColumnTransformer(
    transformers=[(
        'onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        ['transaction_Amount', 'Transaction_Type', 'Location','Device_Type']
    )],
    remainder='passthrough'
)

newX = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test,model.predict(X_test))
print(f"model accuaracy {accuracy}")

joblib.dump(model,"isFraud_model.pkl")

joblib.dump(ct,"isFraud_encoder.pkl")

print("model and encoder saved")

Tranction_amount = float(input("Enter transaction amount: "))
Transaction_type = input("Enter transaction_type (e.g., transfer, payment, withdraw, deposit): ")
Transaction_Time = float(input("Enter time of transaction"))
location = input("Enter location (e.g., ZA, UK, CA, AU, IN): ")
Device_Type = input("Enter device_type (e.g, mobile, web, ATM): ")

predict_data = pd.DataFrame({
    'Transaction_Amount':[Transaction_Amount],
    'Transaction_Type': [Transaction_Type.lower()],
    'Transaction_Time':[Transaction_Time],
    'location': [location.upper()],
    'Device_Type':[Device_Type.lower()]
    })

predictX = ct.transform(predict_data)

prediction = model.predict(predictX)

if prediction[0] == 1:
    print("FRAUD IDENTIFIED ")
else:
    print("valid Transaction")



