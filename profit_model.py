import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

store = pd.read_csv('numeric_copy.csv', encoding="ISO-8859-1")
for col in ['Sales', 'Shipping Cost', 'Discount',  'Profit','shipping_per_item','Order Value']:
    store[col] = (
        store[col]
        .astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)  # remove $, %, commas, etc.
    )
    store[col] = pd.to_numeric(store[col], errors='coerce')


x=store[['Sales','Shipping Cost','Discount','Order Value','shipping_per_item']]
y=store[['Profit']]

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=25, random_state=42, max_depth=6)
rf.fit(x_train, y_train)
import pickle

with open('profit_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
