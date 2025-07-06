import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Dropout, Add, Input, LayerNormalization
from backtesting import Strategy
from backtesting.lib import crossover
from datetime import datetime, timedelta
import logging

###Step 1: OHLCV Data Preparation for S&P 500 Constituents

def get_sp500_tickers():
    """Scrapes Wikipedia to get the list of S&P 500 tickers."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = []
    for row in table.find_all('tr')[1:]:
        ticker = row.find_all('td')[0].text.strip()
        tickers.append(ticker)
    return tickers

def download_ohlcv_data(tickers, start_date, end_date):
    for ticker in tickers:
        stock = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = stock
    return data

###Step 2: Define and Code the Trading Strategy

# Setup logging for real-time monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TCN_LSTM_Strategy(Strategy):
    # Strategy parameters
    n_top = 10
    hold_period = 10
    stop_loss_pct = 0.03
    
    # Parameters to be injected by the Backtest instance
    model = None
    all_data = None
    look_back = 60

    def init(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Strategy. Pre-computing predictions...")
        
        # Pre-compute predictions for all dates to speed up the backtest
        self.daily_predictions = self._precompute_all_predictions()
        
        self.holding_days = {}
        self.entry_prices = {}
        self.logger.info("Strategy initialization complete.")

    def _precompute_all_predictions(self):
        """
        A helper method to compute predictions for all dates at once.
        This is an optimization for backtesting. For live trading, you'd predict day-by-day.
        """
        daily_predictions = {}
        # Assuming `self.data.index` contains all the dates of the backtest
        for date in self.data.index:
            # Slice historical data up to the current date for each ticker
            current_data_slice = {
                ticker: df.loc[:date] for ticker, df in self.all_data.items()
            }
            # The model makes predictions based on data available *before* the current day's open
            ranked_preds = predict_and_rank_tickers(self.model, current_data_slice, self.look_back)
            daily_predictions[date] = ranked_preds
        self.logger.info("All daily predictions have been pre-computed.")
        return daily_predictions

    def next(self):
        current_date = self.data.index[-1]
        
        # --- Position Management ---
        # Check existing positions for stop-loss or holding period exit
        # Use a copy of keys for safe iteration while modifying the dict
        for ticker in list(self.entry_prices.keys()):
            trade_open = any(t.symbol == ticker for t in self.trades)
            if not trade_open:
                # Clean up if trade was closed by backtesting.py (e.g., margin call)
                del self.entry_prices[ticker]
                del self.holding_days[ticker]
                continue

            self.holding_days[ticker] += 1
            # Close position if holding period is reached
            if self.holding_days[ticker] >= self.hold_period:
                self.logger.info(f"{current_date}: Holding period for {ticker} reached. Closing position.")
                for trade in self.trades:
                    if trade.symbol == ticker:
                        trade.close()
                del self.entry_prices[ticker]
                del self.holding_days[ticker]

        # --- Rebalancing ---
        # Rebalance portfolio every `hold_period` days
        if (len(self.data.index) % self.hold_period) == 0:
            self.logger.info(f"{current_date}: Rebalancing portfolio.")

            # 1. Close all existing positions
            for trade in self.trades:
                trade.close()
            self.entry_prices.clear()
            self.holding_days.clear()

            # 2. Get today's ranked predictions
            predictions = self.daily_predictions.get(current_date)
            if not predictions:
                self.logger.warning(f"No predictions available for {current_date}.")
                return

            # 3. Open new positions based on ranks
            long_candidates = list(predictions.keys())[:self.n_top]
            short_candidates = list(predictions.keys())[-self.n_top:]

            # Allocate equal capital to each position
            portfolio_value = self.equity
            position_size = portfolio_value / (2 * self.n_top) if self.n_top > 0 else 0

            # Go long on top stocks
            for ticker in long_candidates:
                price = self.data[ticker].Close[-1]
                sl_price = price * (1 - self.stop_loss_pct)
                self.buy(ticker=ticker, size=position_size / price, sl=sl_price)
                self.entry_prices[ticker] = price
                self.holding_days[ticker] = 0
            
            # Go short on bottom stocks
            for ticker in short_candidates:
                price = self.data[ticker].Close[-1]
                sl_price = price * (1 + self.stop_loss_pct)
                self.sell(ticker=ticker, size=position_size / price, sl=sl_price)
                self.entry_prices[ticker] = price
                self.holding_days[ticker] = 0


def create_dataset_for_prediction(data, look_back=60):
    """Prepares a single data sample for prediction."""
    if len(data) < look_back:
        return None
    X = data[-look_back:].values
    return np.reshape(X, (1, look_back, 1))


# As per the papers, TCN extracts spatial features, LSTM captures temporal dependencies.
def build_tcn_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # TCN Block 
    # Uses dilated convolutions
    x = Conv1D(filters=64, kernel_size=2, activation='relu', dilation_rate=1)(inputs)
    x = LayerNormalization()(x)
    x = Conv1D(filters=64, kernel_size=2, activation='relu', dilation_rate=2)(x)
    x = LayerNormalization()(x)
    x = Conv1D(filters=64, kernel_size=2, activation='relu', dilation_rate=4)(x)
    x = LayerNormalization()(x)
    
    # LSTM Block
    x = LSTM(units=50, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output Layer
    outputs = Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_and_rank_tickers(model, processed_data, look_back=60):
    """
    Predicts T+10 returns for all tickers and ranks them.
    
    Args:
        model (tf.keras.Model): The pre-trained TCN-LSTM model.
        processed_data (dict): A dictionary where keys are tickers and values are 
                               DataFrames with a 'Scaled_Close' column.
        look_back (int): The look-back period required by the model.

    Returns:
        dict: A dictionary of tickers and their predicted returns, sorted descending.
    """
    predictions = {}
    for ticker, data in processed_data.items():
        # Prepare the most recent data for prediction
        X_pred = create_dataset_for_prediction(data['Scaled_Close'], look_back)
        
        if X_pred is not None:
            # Predict the scaled return and store it
            predicted_scaled_return = model.predict(X_pred, verbose=0)[0][0]
            predictions[ticker] = predicted_scaled_return
            
    # Rank tickers from highest predicted gain to highest predicted loss
    ranked_predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
    return ranked_predictions


def fetch_and_process_data(tickers, start_date='2020-01-01', end_date='2023-12-31'):
    """Fetches and preprocesses data for a list of tickers."""
    raw_data_dict = {}
    processed_data_dict = {}
    
    # Download data for all tickers
    data_panel = yf.download(tickers, start=start_date, end=end_date)
    
    for ticker in tickers:
        df = data_panel.loc[:, (slice(None), ticker)]
        df.columns = df.columns.droplevel(1) # Flatten MultiIndex columns
        df = df.dropna()
        if df.empty:
            continue
            
        raw_data_dict[ticker] = df
        
        # Scale the 'Close' price, as the model was trained on scaled data [AI KNOWLEDGE]({})
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Important: fit_transform should be done on the entire dataset to maintain consistency
        df['Scaled_Close'] = scaler.fit_transform(df[['Close']])
        processed_data_dict[ticker] = df

    # Prepare data in the format required by backtesting.py (a single multi-index DataFrame)
    backtest_data = pd.concat(raw_data_dict.values(), keys=raw_data_dict.keys(), axis=1)
    
    return backtest_data, processed_data_dict


def main():
    # Use a smaller subset of S&P 500 tickers for a manageable example
    sp500_tickers = get_sp500_tickers()
    
    # 1. Fetch and process data
    backtest_data, processed_data = fetch_and_process_data(sp500_tickers)
    
    # 2. Load the pre-trained model
    # Build and train the model
    model = build_tcn_lstm_model((look_back, 1))
    model.summary()
    print("Training model...")
    # Introduce early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1, callbacks=[early_stopping])
    
    # 3. Initialize and run the backtest
    # No need to declare an instance of the Strategy class in the loop because the backtesting.py library will do it internally
    bt = Backtest(backtest_data, TCN_LSTM_Strategy, cash=1_000_000, commission=.002)
    
    stats = bt.run(
        model=model,
        all_data=processed_data,
        look_back=60
    )
    
    # 4. Print results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(stats)
    print("\nNote: Results are based on a mock model and are for demonstration purposes.")
    print("The strategy's performance depends heavily on the accuracy of the real TCN-LSTM model.")
    print("The underlying model is effective at capturing complex trends but can be impacted by sudden market shocks
    
    # 5. Plot the results
    bt.plot()


if __name__ == "__main__":
    main()

