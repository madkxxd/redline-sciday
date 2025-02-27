import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the cleaned dataset
df = pd.read_csv("cleaned_blood_donation.csv")

# Check for missing values and handle them
df.fillna(method='ffill', inplace=True)

# Selecting features relevant to time-series prediction
data = df['Total_Volume_Normalized'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Creating sequences for LSTM
def create_sequences(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10
X, y = create_sequences(data_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("lstm_blood_donation.h5")

# Donor Matching Model (KNN)
# Assuming a synthetic donor dataset (please provide actual donor dataset if available)
donor_data = pd.DataFrame({
    'Latitude': np.random.uniform(-90, 90, size=len(df)),
    'Longitude': np.random.uniform(-180, 180, size=len(df)),
    'Blood_Type': np.random.randint(1, 5, size=len(df)),
    'Availability': np.random.randint(0, 2, size=len(df))
})

X_donors = donor_data[['Latitude', 'Longitude', 'Blood_Type']]
y_donors = donor_data['Availability']

# Feature Scaling for KNN
scaler_knn = StandardScaler()
X_donors_scaled = scaler_knn.fit_transform(X_donors)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_donors_scaled, y_donors, test_size=0.2, random_state=42)

# Finding optimal k-value
scores = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train_d, y_train_d, cv=5).mean()
    scores.append(score)

best_k = np.argmax(scores) + 1  # Index starts at 0
logging.info(f"Best k: {best_k}")

# Train the optimized KNN model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_d, y_train_d)
accuracy = knn.score(X_test_d, y_test_d)
logging.info(f"Donor Matching Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained KNN model
joblib.dump(knn, "knn_donor_matching.pkl")
