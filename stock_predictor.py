import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class StockPricePredictor:
    def __init__(self, symbol='AAPL', period='5y'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.model = None
        self.scaler = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.sequence_length = 60
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        print(f"📈 Fetching data for {self.symbol}...")
        
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
                
            print(f"✅ Data fetched successfully!")
            print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            print(f"Total data points: {len(self.data)}")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def add_technical_indicators(self):
        """Add technical indicators as features"""
        df = self.data.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / df['BB_width']
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        self.data = df
        return df
    
    def prepare_data(self, target_column='Close', test_size=0.2):
        """Prepare data for LSTM training"""
        print("🔧 Preparing data for training...")
        
        # Add technical indicators
        self.add_technical_indicators()
        
        # Select features
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'RSI', 'BB_position',
            'Volatility', 'Price_Change', 'High_Low_Pct', 'Volume_ratio'
        ]
        
        # Remove rows with NaN values
        df_clean = self.data[feature_columns].dropna()
        
        # Scale the data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(df_clean)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, feature_columns.index(target_column)])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        print(f"✅ Data prepared!")
        print(f"Training sequences: {len(self.X_train)}")
        print(f"Test sequences: {len(self.X_test)}")
        print(f"Features: {len(feature_columns)}")
        print(f"Sequence length: {self.sequence_length}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def build_lstm_model(self, lstm_units=50, dropout_rate=0.2, num_layers=2):
        """Build LSTM model"""
        print("🏗️ Building LSTM model...")
        
        model = Sequential()
        
        # First LSTM layer
        if num_layers > 1:
            model.add(Bidirectional(LSTM(lstm_units, return_sequences=True), 
                                  input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        else:
            model.add(Bidirectional(LSTM(lstm_units), 
                                  input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, num_layers):
            if i == num_layers - 1:  # Last layer
                model.add(Bidirectional(LSTM(lstm_units // (i+1))))
            else:
                model.add(Bidirectional(LSTM(lstm_units // (i+1), return_sequences=True)))
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mean_squared_error',
                     metrics=['mae'])
        
        self.model = model
        print("✅ Model built successfully!")
        return model
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.1):
        """Train the LSTM model"""
        print("🚀 Training model...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return history
    
    def make_predictions(self):
        """Make predictions on test data"""
        print("🔮 Making predictions...")
        
        # Predict on test data
        y_pred_scaled = self.model.predict(self.X_test)
        
        # Create dummy array for inverse scaling
        dummy_array = np.zeros((len(y_pred_scaled), self.scaler.n_features_in_))
        dummy_array[:, 3] = y_pred_scaled.flatten()  # Close price is at index 3
        y_pred = self.scaler.inverse_transform(dummy_array)[:, 3]
        
        # Inverse scale actual values
        dummy_array_actual = np.zeros((len(self.y_test), self.scaler.n_features_in_))
        dummy_array_actual[:, 3] = self.y_test
        y_actual = self.scaler.inverse_transform(dummy_array_actual)[:, 3]
        
        return y_actual, y_pred
    
    def calculate_metrics(self, y_actual, y_pred):
        """Calculate prediction metrics"""
        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        
        # Directional accuracy
        actual_direction = np.diff(y_actual) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        print("📊 Model Performance Metrics:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Directional Accuracy: {directional_accuracy:.2%}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def plot_predictions(self, y_actual, y_pred, days_to_show=200):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(15, 8))
        
        # Show only last N days for clarity
        if len(y_actual) > days_to_show:
            y_actual_plot = y_actual[-days_to_show:]
            y_pred_plot = y_pred[-days_to_show:]
        else:
            y_actual_plot = y_actual
            y_pred_plot = y_pred
        
        plt.plot(y_actual_plot, label='Actual Price', linewidth=2, alpha=0.8)
        plt.plot(y_pred_plot, label='Predicted Price', linewidth=2, alpha=0.8)
        
        plt.title(f'{self.symbol} Stock Price Prediction', fontsize=16, fontweight='bold')
        plt.xlabel('Time (Days)', fontsize=12)
        plt.ylabel('Stock Price ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add metrics as text
        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        directional_accuracy = np.mean(np.diff(y_actual) > 0 == np.diff(y_pred) > 0)
        
        plt.text(0.02, 0.98, f'RMSE: ${rmse:.2f}\nDirectional Accuracy: {directional_accuracy:.1%}',
                transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_future(self, days=30):
        """Predict future stock prices"""
        print(f"🔮 Predicting next {days} days...")
        
        # Get the last sequence from training data
        last_sequence = self.X_test[-1].reshape(1, self.sequence_length, -1)
        future_predictions = []
        
        for _ in range(days):
            # Predict next value
            next_pred = self.model.predict(last_sequence, verbose=0)
            future_predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            # Create new row with predicted value
            new_row = last_sequence[0, -1].copy()
            new_row[3] = next_pred[0, 0]  # Update Close price
            
            # Shift sequence and add new row
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1] = new_row
        
        # Inverse scale predictions
        dummy_array = np.zeros((len(future_predictions), self.scaler.n_features_in_))
        dummy_array[:, 3] = future_predictions
        future_prices = self.scaler.inverse_transform(dummy_array)[:, 3]
        
        return future_prices
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"📁 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"📁 Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    print("📈 Stock Price Prediction with LSTM")
    print("=" * 50)
    
    # Initialize predictor
    predictor = StockPricePredictor(symbol='AAPL', period='5y')
    
    # Fetch and prepare data
    data = predictor.fetch_data()
    if data is not None:
        X_train, y_train, X_test, y_test = predictor.prepare_data()
        
        # Build and train model
        model = predictor.build_lstm_model(lstm_units=100, num_layers=3)
        history = predictor.train_model(epochs=50, batch_size=32)
        
        # Make predictions and evaluate
        y_actual, y_pred = predictor.make_predictions()
        metrics = predictor.calculate_metrics(y_actual, y_pred)
        
        # Plot results
        predictor.plot_training_history(history)
        predictor.plot_predictions(y_actual, y_pred)
        
        # Predict future prices
        future_prices = predictor.predict_future(days=30)
        print(f"\n🔮 Predicted price in 30 days: ${future_prices[-1]:.2f}")
        
        # Save model
        predictor.save_model('stock_prediction_model.h5')
        
        print("✅ Stock prediction completed!")
