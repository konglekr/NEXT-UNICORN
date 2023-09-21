import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.load_weights('weights.h5')

# Define the ticker symbol and date range
ticker_symbol = ['googl', 'mo', 'cvx', 'xom', 'mcd', 'tsla']
for ticker in ticker_symbol:
    start_date = '2015-01-01'
    end_date = '2023-09-20'

    # Fetch historical stock data
    df = yf.download(ticker, start=start_date, end=end_date)

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
    train_size = 0
    _, X_test, _, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Denormalize the predictions
    predictions = scaler.inverse_transform(predictions)

    y_orig = scaler.inverse_transform(y)

    # Implement a basic trading logic (Sample logic: Buy when predicted price is higher, sell when it's lower)
    initial_balance = 10000  # Replace with your initial balance
    balance = initial_balance
    position = 0

    buy_prices = []  # To track buy prices
    sell_prices = []  # To track sell prices

    for i in range(len(predictions)):
        if predictions[i] > y_orig[i][0]:  # Buy signal
            position += balance / y_orig[i][0]
            balance = 0
            buy_prices.append(y_orig[i][0])
        elif predictions[i] < y_orig[i][0]:  # Sell signal
            balance += position * y_orig[i][0]
            position = 0
            sell_prices.append(y_orig[i][0])

    # Calculate final balance and trading bot metrics
    final_balance = balance + (position * df['Close'][-1])
    total_return_bot = (final_balance - initial_balance) / initial_balance

    # Calculate buy-and-hold metrics
    initial_price = df['Close'][0]
    final_price = df['Close'][-1]
    total_return_buy_hold = (final_price - initial_price) / initial_price

    # Print metrics
    print(f'Total Return (Trading Bot - {ticker}): {total_return_bot:.4f}')
    print(f'Total Return (Buy and Hold - {ticker}): {total_return_buy_hold:.4f}')

    #plt.scatter(predictions, df['Close'][60:])
    #plt.show()
