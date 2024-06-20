import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib

# Function to create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]  # Predicting 'Adj Close'
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Streamlit app
st.title("Stock Price Prediction")

# User inputs
ticker = st.text_input("Enter stock ticker:", value="IOB.BO")
start_date = st.date_input("Start date:", value=pd.to_datetime('2003-09-04'))
end_date = st.date_input("End date:", value=pd.to_datetime('2024-06-07'))

if st.button("Predict"):
    with st.spinner("Downloading data..."):
        # Download historical stock price data
        data = yf.download(ticker, start=start_date, end=end_date)
    
    st.write(data.head())

    # Select the 'Adj Close' column for prediction and create lagged features
    data['Lag1'] = data['Adj Close'].shift(1)
    data['Lag2'] = data['Adj Close'].shift(2)
    data['Lag3'] = data['Adj Close'].shift(3)
    data['Lag4'] = data['Adj Close'].shift(4)
    data['Lag5'] = data['Adj Close'].shift(5)
    
    # Drop the rows with NaN values after shifting
    data = data.dropna()
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Adj Close', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5']])
    
    # Create sequences
    seq_length = 60  # Number of previous days to consider for predicting the next day
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build the LSTM model
    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(seq_length, X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    # Train the model
    with st.spinner("Training model..."):
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])
    
    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    st.write(f'Loss (Mean Squared Error): {loss}')
    
    # Plot training & validation loss values
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Validation'], loc='upper right')
    st.pyplot(fig)
    
    # Prepare the latest data for prediction
    latest_data = scaled_data[-seq_length:]
    latest_data = latest_data.reshape((1, seq_length, X.shape[2]))
    
    # Predict the next day's closing price
    predicted_scaled = model.predict(latest_data)
    
    # Ensure the predicted shape matches the scaler's expected input shape
    predicted_scaled = np.reshape(predicted_scaled, (1, 1))
    
    # Create an array with the same shape as the scaler's fit input to inverse transform correctly
    # Only the first column is used for 'Adj Close', the rest can be filled with zeros or the last row's values
    inverse_transform_input = np.zeros((1, scaled_data.shape[1]))
    inverse_transform_input[0, 0] = predicted_scaled[0, 0]
    
    # Inverse transform the predicted scaled value
    predicted_price = scaler.inverse_transform(inverse_transform_input)
    
    st.write(f'Predicted next day adjusted close price: {predicted_price[0][0]}')
