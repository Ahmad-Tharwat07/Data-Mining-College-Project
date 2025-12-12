# https://www.kaggle.com/datasets/abdelrahmanahmed110/used-cars-for-sale-in-egypt/data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'hatla2ee_scraped_data.csv')
data = pd.read_csv(data_path)
df = data.copy()
df.head()

data.info()

df.isnull().sum()

df.describe()

df.drop_duplicates(inplace=True)

# to remove EGP and KM from Price and Mileage
def clean_num(x):
    if type(x) == str:
        return float(x.split()[0].replace(',' , ''))
    else:
        return np.nan
df["Price"] = df["Price"].apply(clean_num)
df["Mileage"] = df["Mileage"].apply(clean_num)

df.head()

print(len(df))

df = df.dropna(subset=['Price']).copy() # Remove missing targets and ensure it's a new copy
df["Mileage"] = df["Mileage"].fillna(df["Mileage"].median()) # Fill missing values with median

print(len(df))

df.isnull().sum() # GG =))

df['Model_Year'] = (df["Name"].apply(lambda x: x.split()[-1])).astype("int") # get model year from Name column.

df.isnull().sum()

df['Model_Year'].unique()

df.drop(df[df["Model_Year"] == '0'].index,inplace=True)

# Create Car_Age Feature
df['Date Displayed'] = pd.to_datetime(df['Date Displayed'])
df['Car_Age'] = df['Date Displayed'].dt.year - df['Model_Year']

df['Car_Age'].unique()

df.loc[df['Car_Age'] < 0, 'Car_Age'] = 0
df.loc[df['Car_Age'] == 2024, 'Car_Age'] = df['Car_Age'].median()

df['Car_Age'].unique()

df.describe()
# min price is 1060??

sorted_values = df.sort_values(by='Price', ascending=True)
sorted_values.head()

print(len(df))
df = df[df['Price'] > 5000] # keep what's greater than 5000 EGP
print(len(df))

for col in ["Mileage","Price"]:
    sns.histplot(data=df, x = col, kde=True)
    plt.show()

# log transformation
df['Price'] = np.log1p(df['Price'])
df['Mileage'] = np.log1p(df['Mileage'])

encoders = {}
label_encoder_columns = ['Make', 'Model']

for col in label_encoder_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

binary_cols = ['Automatic Transmission', 'Air Conditioner', 'Power Steering', 'Remote Control']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

selected_features = ['Mileage', 'Make', 'Model', 'Automatic Transmission', 'Air Conditioner', 'Power Steering', 'Remote Control', 'Car_Age']
X = df[selected_features]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

R2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R-squared score: {R2:.4f}")
print(f"Mean Absolute Error (MAE): {MAE:,.2f}")
print(f"Root Mean Squared Error (RMSE): {RMSE:,.2f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='black', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Line')

plt.xlabel('Actual Price (EGP)')
plt.ylabel('Predicted Price (EGP)')
plt.title(f'Actual vs Predicted Prices (R2: {R2:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('random_forest_label_standard_predictions.png')
plt.show()

def predict_from_input():
    print("\n--- Enter Car Details ---")

    try:
        make = input("Enter Make (e.g. Kia): ").strip()
        model_name = input("Enter Model (e.g. Sportage): ").strip()

        # Check if the model knows this Make/Model
        if make not in encoders['Make'].classes_:
            print(f"\n Error: The model doesn't know the Make '{make}'.")
            print(f"Try one of these: {encoders['Make'].classes_[:5]}")
            return

        if model_name not in encoders['Model'].classes_:
            print(f"\nError: The model doesn't know the Model '{model_name}'.")
            return
        # ----------------------------------

        mileage_raw = float(input("Enter Mileage (in KM): "))
        made_year = int(input("Enter Made Year (e.g. 2020): "))
        listing_year = int(input("Enter Listing Year (e.g. 2024): "))

        print("For the following, answer 'y' for Yes or 'n' for No:")
        auto = 1 if input("Automatic Transmission? ").lower() == 'y' else 0
        ac = 1 if input("Air Conditioner? ").lower() == 'y' else 0
        power_steer = 1 if input("Power Steering? ").lower() == 'y' else 0
        remote = 1 if input("Remote Control? ").lower() == 'y' else 0

        # 2. Process Data
        mileage_log = np.log1p(mileage_raw)
        car_age = max(0, listing_year - made_year)

        # Encode
        make_encoded = encoders['Make'].transform([make])[0]
        model_encoded = encoders['Model'].transform([model_name])[0]

        # 3. Create DataFrame
        input_data = pd.DataFrame([{
            'Mileage': mileage_log,
            'Make': make_encoded,
            'Model': model_encoded,
            'Automatic Transmission': auto,
            'Air Conditioner': ac,
            'Power Steering': power_steer,
            'Remote Control': remote,
            'Car_Age': car_age
        }])

        # Ensure correct column order
        input_data = input_data[X.columns]

        # 4. Scale and Predict
        input_scaled = scaler.transform(input_data)
        log_price = model.predict(input_scaled)[0]
        real_price = np.expm1(log_price)

        print("\n" + "="*40)
        print(f"Predicted Price: {real_price:,.0f} EGP")
        print("="*40)

    except Exception as e:
        print(f"\n Error: {e}")

predict_from_input()
