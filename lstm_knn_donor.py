import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("cleaned_blood_donation.csv")

# Selecting features relevant to time-series prediction
data = df['Total_Volume_Normalized'].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

# Creating sequences for LSTM
def create_sequences(data, time_step=10):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i+time_step, 0])
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
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

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
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_donors, y_donors, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_d, y_train_d)
accuracy = knn.score(X_test_d, y_test_d)
print(f"Donor Matching Model Accuracy: {accuracy * 100:.2f}%")
