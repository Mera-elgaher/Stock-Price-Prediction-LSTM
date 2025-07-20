import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not installed. Model training will not be available.")
    TF_AVAILABLE = False

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
        if self.data is None:
            print("❌ No data available. Fetch data first.")
            return None
            
        df = self.data.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        
        self.data = df
        return df
    
    def prepare_data(self, target_column='Close', test_size=0.2):
        """Prepare data for LSTM training"""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available. Cannot prepare data for training.")
            return None, None, None, None
            
        print("🔧 Preparing data for training...")
        
        # Add technical indicators
        self.add_technical_indicators()
        
        # Select features
        feature_columns = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'Volatility', 'Price_Change']
        
        # Remove rows with NaN values
        df_clean = self.data[feature_columns].dropna()
        
        if len(df_clean) < self.sequence_length + 10:
            print(f"❌ Not enough data. Need at least {self.sequence_length + 10} rows, got {len(df_clean)}")
            return None, None, None, None
        
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
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available. Cannot build model.")
            return None
            
        if self.X_train is None:
            print("❌ No training data. Run prepare_data() first.")
            return None
            
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
    
    def train_model(self, epochs=50, batch_size=32, validation_split=0.1):
        """Train the LSTM model"""
        if not TF_AVAILABLE or self.model is None:
            print("❌ Model not available. Build model first.")
            return None
            
        print("🚀 Training model...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
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
        if self.model is None:
            print("❌ Model not trained. Train model first.")
            return None, None
            
        print("🔮 Making predictions...")
        
        # Predict on test data
        y_pred_scaled = self.model.predict(self.X_test)
        
        # Create dummy array for inverse scaling
        dummy_array = np.zeros((len(y_pred_scaled), self.scaler.n_features_in_))
        dummy_array[:, 0] = y_pred_scaled.flatten()  # Close price is at index 0
        y_pred = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        # Inverse scale actual values
        dummy_array_actual = np.zeros((len(self.y_test), self.scaler.n_features_in_))
        dummy_array_actual[:, 0] = self.y_test
        y_actual = self.scaler.inverse_transform(dummy_array_actual)[:, 0]
        
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
    
    def plot_data(self, days=100):
        """Plot recent stock data"""
        if self.data is None:
            print("❌ No data to plot. Fetch data first.")
            return
            
        recent_data = self.data.tail(days)
        
        plt.figure(figsize=(12, 6))
        plt.plot(recent_data.index, recent_data['Close'], label='Close Price', linewidth=2)
        
        if 'SMA_20' in recent_data.columns:
            plt.plot(recent_data.index, recent_data['SMA_20'], label='SMA 20', alpha=0.7)
        if 'SMA_50' in recent_data.columns:
            plt.plot(recent_data.index, recent_data['SMA_50'], label='SMA 50', alpha=0.7)
        
        plt.title(f'{self.symbol} Stock Price - Last {days} Days', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def get_latest_price(self):
        """Get the latest stock price"""
        if self.data is None:
            self.fetch_data()
        
        if self.data is not None:
            latest_price = self.data['Close'].iloc[-1]
            latest_date = self.data.index[-1].date()
            print(f"📊 {self.symbol} Latest Price: ${latest_price:.2f} (as of {latest_date})")
            return latest_price, latest_date
        return None, None
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            print("❌ No model to save.")
            return False
            
        try:
            self.model.save(filepath)
            print(f"📁 Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

# Example usage and demo
if __name__ == "__main__":
    print("📈 Stock Price Prediction with LSTM")
    print("=" * 50)
    
    # Initialize predictor
    predictor = StockPricePredictor(symbol='AAPL', period='2y')
    
    # Fetch data
    data = predictor.fetch_data()
    
    if data is not None:
        # Show latest price
        predictor.get_latest_price()
        
        # Plot recent data
        print("\n📊 Plotting recent stock data...")
        predictor.plot_data(days=90)
        
        # Add technical indicators
        print("\n🔧 Adding technical indicators...")
        predictor.add_technical_indicators()
        print("✅ Technical indicators added!")
        
        if TF_AVAILABLE:
            # Prepare data for training
            print("\n📋 Preparing data for training...")
            X_train, y_train, X_test, y_test = predictor.prepare_data()
            
            if X_train is not None:
                # Build model
                print("\n🏗️ Building LSTM model...")
                model = predictor.build_lstm_model(lstm_units=100, num_layers=3)
                
                print("\n✅ Stock predictor ready!")
                print("🚀 Next steps:")
                print("1. Run predictor.train_model() to train the model")
                print("2. Run predictor.make_predictions() to get predictions")
                print("3. Run predictor.calculate_metrics() to evaluate performance")
            else:
                print("❌ Failed to prepare data for training")
        else:
            print("\n💡 Install TensorFlow to enable model training:")
            print("pip install tensorflow")
    else:
        print("❌ Failed to fetch stock data")
