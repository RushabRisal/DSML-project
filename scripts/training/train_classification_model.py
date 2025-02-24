import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
dataframe = pd.read_csv('./data/cleanDataAll.csv')

# Prepare dataset
dataframe['Price_Fluctuation'] = dataframe['Average'].diff().fillna(0)
dataframe['Price_Fluctuation'] = dataframe['Price_Fluctuation'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

# Features and target
X = dataframe[['Average']]
y = dataframe['Price_Fluctuation']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Save model
joblib.dump(model, './models/price_fluctuation_model.joblib')