import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)
print("DEBUG: Flask app initialized")

# --------------------
# Health Check
# --------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Stock Prediction API is running"
    })

# --------------------
# Internal Imports
# --------------------
from services.prediction import (
    train_linear_regression,
    predict_future_linear,
    train_lstm_model,
    predict_future_lstm,
    calculate_trading_signals
)

from services.coingecko import (
    is_crypto_symbol,
    fetch_crypto_historical_data,
    fetch_crypto_current_price,
    get_crypto_info
)

from services.binance_api import (
    get_binance_klines,
    get_binance_price,
    get_binance_24hr_stats
)

# Optional TradingView import (prevents Render crashes)
try:
    from tradingview_ta import TA_Handler, Interval
    TRADINGVIEW_AVAILABLE = True
except Exception:
    TRADINGVIEW_AVAILABLE = False
    TA_Handler = None
    Interval = None


def fetch_full_stock_data(symbol):
    symbol = symbol.upper()
    is_crypto = is_crypto_symbol(symbol)
    
    hist = None
    data_source = None
    crypto_info = {"name": symbol}

    if is_crypto or symbol in ('XAUUSD', 'GOLD'):
        # -------- GOLD (XAUUSD) --------
        if symbol in ('XAUUSD', 'GOLD'):
            crypto_info = {'name': 'Gold Spot (XAU/USD)'}
            # yfinance fallback (IAU proxy)
            stock = yf.Ticker("IAU")
            hist = stock.history(period="1y")
            if not hist.empty:
                hist.reset_index(inplace=True)
                scale_factor = 53.4
                for col in ['Open', 'High', 'Low', 'Close']:
                    hist[col] *= scale_factor
                data_source = "yfinance IAU (Scaled)"
        # -------- CRYPTO --------
        else:
            hist = get_binance_klines(symbol)
            if hist is not None and not hist.empty:
                data_source = "Binance API"
                crypto_info = {"name": f"{symbol}/USDT"}
            else:
                hist = fetch_crypto_historical_data(symbol, days=365)
                crypto_info = get_crypto_info(symbol)
                data_source = "CoinGecko"
    else:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        if not hist.empty:
            hist.reset_index(inplace=True)
            data_source = "yfinance"

    return hist, data_source, crypto_info

@app.route('/api/price/<symbol>', methods=['GET'])
def get_live_price(symbol):
    try:
        symbol = symbol.upper()
        if is_crypto_symbol(symbol):
            # Check Binance first
            price = get_binance_price(symbol)
            if price:
                return jsonify({"price": price})
            # Fallback to CoinGecko
            price_data = fetch_crypto_current_price(symbol)
            if price_data:
                return jsonify({"price": price_data['price']})
        elif symbol in ('XAUUSD', 'GOLD'):
            # Gold price logic from get_stock_data
            paxg_price = get_binance_price("PAXGUSDT")
            if paxg_price:
                return jsonify({"price": paxg_price})
            # fallback via IAU
            stock = yf.Ticker("IAU")
            data = stock.history(period='1d')
            if not data.empty:
                return jsonify({"price": float(data['Close'].iloc[-1]) * 53.4})
        else:
            stock = yf.Ticker(symbol)
            data = stock.history(period='1d')
            if not data.empty:
                return jsonify({"price": float(data['Close'].iloc[-1])})
        
        return jsonify({"error": "Price not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/<symbol>', methods=['GET'])
def get_stock_prediction(symbol):
    model_type = request.args.get('model', 'linear')
    try:
        hist, _, _ = fetch_full_stock_data(symbol)
        if hist is None or hist.empty:
            return jsonify({"error": "No data found for prediction"}), 404
        
        # Ensure RSI is there for signals/matching data if needed
        delta = hist['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        predictions = []
        if model_type == 'linear':
            model = train_linear_regression(hist)
            predictions = predict_future_linear(model, hist.iloc[-1]['Date'])
        else:
            model, scaler, scaled_data = train_lstm_model(hist)
            future_prices = predict_future_lstm(model, scaler, scaled_data)
            
            last_date = pd.to_datetime(hist.iloc[-1]['Date'])
            for i, p in enumerate(future_prices):
                future_date = last_date + pd.Timedelta(days=i+1)
                predictions.append({
                    "date": future_date.strftime('%Y-%m-%d'),
                    "price": float(p)
                })

        return jsonify({
            "symbol": symbol,
            "model": model_type,
            "predictions": predictions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        symbol = symbol.upper()
        hist, data_source, crypto_info = fetch_full_stock_data(symbol)

        if hist is None or hist.empty:
            return jsonify({"error": "No data found"}), 404

        # Clean Dates
        if 'Date' in hist.columns:
            hist['Date'] = pd.to_datetime(hist['Date']).dt.strftime('%Y-%m-%d')

        # =========================
        # INDICATORS
        # =========================
        hist['SMA_20'] = hist['Close'].rolling(20).mean()
        hist['SMA_50'] = hist['Close'].rolling(50).mean()

        delta = hist['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        df = hist.dropna()
        data = [{
            "Date": r['Date'],
            "Open": float(r['Open']),
            "High": float(r['High']),
            "Low": float(r['Low']),
            "Close": float(r['Close']),
            "Volume": int(r['Volume']),
            "SMA_20": float(r['SMA_20']),
            "SMA_50": float(r['SMA_50']),
            "RSI": float(r['RSI'])
        } for _, r in df.iterrows()]

        latest = df.iloc[-1]
        stats = {
            "open": float(latest['Open']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "close": float(latest['Close']),
            "volume": int(latest['Volume'])
        }

        signals = calculate_trading_signals(hist)

        return jsonify({
            "symbol": symbol,
            "company": crypto_info["name"],
            "data": data,
            "stats": stats,
            "signals": signals,
            "data_source": data_source,
            "warning": None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# ENTRY POINT (Render)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"DEBUG: Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
