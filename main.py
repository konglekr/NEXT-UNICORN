import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Define the ticker symbol and date range
ticker_symbol = 'spy'  # Replace with the symbol of the stock you want to analyze
start_date = '2010-01-01'
end_date = '2020-08-31'

# Fetch historical stock data
df = yf.download(ticker_symbol, start=start_date, end=end_date)

# Check for missing values and fill or drop them as needed
df.dropna(inplace=True)

dates = df.index.astype(str).to_list()

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Define sequence length and create sequences
sequence_length = 60  # Adjust as needed
sequences = []
next_prices = []

for i in range(len(scaled_data) - sequence_length):
    sequences.append(scaled_data[i:i+sequence_length])
    next_prices.append(scaled_data[i+sequence_length])

X = np.array(sequences)
y = np.array(next_prices)

# Split the data into training and testing sets
train_size = int(len(X) * 0.5)
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=50)

# Make predictions on the test data
predictions = model.predict(X_test)

# Denormalize the predictions
predictions = scaler.inverse_transform(predictions)

# Implement a basic trading logic (Sample logic: Buy when predicted price is higher, sell when it's lower)
initial_balance = 10000  # Replace with your initial balance
balance = initial_balance
position = 0

buy_prices = []  # To track buy prices
sell_prices = []  # To track sell prices

for i in range(len(predictions)):
    if predictions[i] > df['Close'][i + train_size]:  # Buy signal
        position += balance / df['Close'][i + train_size]
        balance = 0
        buy_prices.append(df['Close'][i + train_size])
    elif predictions[i] < df['Close'][i + train_size]:  # Sell signal
        balance += position * df['Close'][i + train_size]
        position = 0
        sell_prices.append(df['Close'][i + train_size])

# Calculate final balance and trading bot metrics
final_balance = balance + (position * df['Close'][-1])
total_return_bot = (final_balance - initial_balance) / initial_balance

# Calculate buy-and-hold metrics
initial_price = df['Close'][train_size]
final_price = df['Close'][-1]
total_return_buy_hold = (final_price - initial_price) / initial_price

# Print metrics
print(f'Total Return (Trading Bot): {total_return_bot:.4f}')
print(f'Total Return (Buy and Hold): {total_return_buy_hold:.4f}')

model.save_weights('weights.h5')
