from stock_predictor import StockPricePredictor

# Example usage
print("📈 Stock Price Prediction Example")
print("=" * 40)

# Initialize predictor for Apple stock
predictor = StockPricePredictor(symbol='AAPL', period='2y')

# Fetch data
print("Fetching stock data...")
data = predictor.fetch_data()

if data is not None:
    print("✅ Data fetched successfully!")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Show latest price
    predictor.get_latest_price()
    
    # Add technical indicators
    print("\nAdding technical indicators...")
    predictor.add_technical_indicators()
    
    # Prepare data for training
    print("\nPreparing data for training...")
    X_train, y_train, X_test, y_test = predictor.prepare_data()
    
    if X_train is not None:
        # Build model
        print("Building LSTM model...")
        model = predictor.build_lstm_model(lstm_units=50, num_layers=2)
        
        if model is not None:
            print("✅ Model ready for training!")
            print("\n🚀 Next steps:")
            print("1. predictor.train_model(epochs=10)")
            print("2. y_actual, y_pred = predictor.make_predictions()")
            print("3. predictor.calculate_metrics(y_actual, y_pred)")
        else:
            print("❌ Failed to build model")
    else:
        print("❌ Failed to prepare data")
else:
    print("❌ Failed to fetch data")
