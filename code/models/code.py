# necessary imports
import numpy as np
import pandas as pd

# from tqdm.notebook import t
from tqdm.keras import TqdmCallback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras.utils import to_categorical


df = pd.read_csv('code\datasets\stardata.csv')

target_variable = 'Spectral Class'
df_encoded = pd.get_dummies(df, columns=['Star color'], dtype=int)

# Separate the features and the target variable for model training
X = df_encoded.drop(['Spectral Class'], axis=1)
y = df_encoded['Spectral Class']

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

input_shape = X_train_scaled.shape[1]
num_classes = len(np.unique(y_train))

inputs = tf.keras.Input(shape=(input_shape,))

x = layers.Dense(100, activation='relu')(inputs)
x = layers.Dense(100, activation='relu')(x)

outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


num_epochs = 50
batch_size = 64

history = model.fit(
    X_train_scaled, y_train_encoded,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(X_test_scaled, y_test_encoded),
    verbose=0,
    callbacks=[TqdmCallback(verbose=1)]
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test_scaled, y_test_encoded)
print(f"Accuracy: {accuracy}")

model.save('model\mymodel.h5')
