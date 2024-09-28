# necessary imports
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os

notebook_path = os.path.abspath("f24-spectral-class-prediction.ipynb")
datafile = os.path.join(os.path.dirname(notebook_path), 'code', 'datasets', 'stardata.csv')

df = pd.read_csv(datafile)
df['Star color'] = df['Star color'].str.lower().str.strip().str.replace('-', ' ')

df_encoded = pd.get_dummies(df, columns=['Star color'], dtype=int)

with open(os.path.join('code', 'models', 'content', 'columns.txt'), 'w') as f:
    f.write(str(df_encoded.columns))

# Separate the features and the target variable for model training
X = df_encoded.drop(['Spectral Class'], axis=1)
y = df_encoded['Spectral Class']

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the standardizer
pickle.dump(scaler, open(os.path.join('models', 'scaler.pkl'), 'wb')) 
pickle.dump(scaler, open(os.path.join('code', 'models', 'content', 'scaler.pkl'), 'wb')) 

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

# Save models
with open(os.path.join('models', 'mymodel.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join('code', 'models', 'content', 'mymodel.pkl'), 'wb') as f:
    pickle.dump(model, f)
