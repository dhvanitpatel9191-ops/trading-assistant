import os
from flask import Flask, render_template, request, jsonify, redirect
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# Optional dependencies
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("Warning: flask-cors not available, CORS disabled")

# Firebase
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    firebase_admin = None
    credentials = None
    firestore = None
    print("Warning: firebase-admin not available, Firebase features disabled")

# Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    print("Warning: google-generativeai not available, AI features disabled")

# Import Python engine
try:
    from python.engine import (
    calculate_volatility,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    find_support_resistance
)

except ImportError as e:
    print(f"CRITICAL: Could not import engine module: {e}")
    raise

# Initialize Firebase
db = None
if FIREBASE_AVAILABLE:
    try:
        service_key_path = "serviceKey.json"
        if not os.path.exists(service_key_path):
            service_key_path = "python/serviceKey.json"
        
        if os.path.exists(service_key_path):
            cred = credentials.Certificate(service_key_path)
            try:
                firebase_admin.initialize_app(cred)
            except ValueError:
                pass
            db = firestore.client()
            print("Firebase initialized successfully")
        else:
            print("Warning: serviceKey.json not found, Firebase disabled")
    except Exception as e:
        print(f"Warning: Could not initialize Firebase: {e}")
        db = None

# Initialize Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini AI configured successfully")
    except Exception as e:
        print(f"Warning: Could not configure Gemini AI: {e}")

# Flask app
app = Flask(__name__, template_folder='templates')
if CORS_AVAILABLE:
    CORS(app)

@app.errorhandler(500)
def internal_error(error):
    print(f"500 Error: {error}")
    import traceback
    traceback.print_exc()
    try:
        return render_template('index.html', error="An internal server error occurred. Please try again."), 500
    except:
        return "Internal Server Error", 500


def analyze_stock(stock_symbol, date_from, date_to):
    """Analyze stock using Python engine and yfinance"""
    try:
        print(f"=== Starting analysis for {stock_symbol} from {date_from} to {date_to} ===")

        # Validate and adjust dates
        try:
            date_from_obj = datetime.strptime(date_from, "%Y-%m-%d").date()
            date_to_obj = datetime.strptime(date_to, "%Y-%m-%d").date()
            today = date.today()

            if date_to_obj > today:
                print(f"INFO: End date {date_to} is in the future, adjusting to today ({today})")
                date_to_obj = today
                date_to = today.strftime("%Y-%m-%d")

            if date_from_obj > today:
                print(f"INFO: Start date {date_from} is in the future, adjusting to 30 days ago")
                date_from_obj = today - timedelta(days=30)
                date_from = date_from_obj.strftime("%Y-%m-%d")

            if date_from_obj > date_to_obj:
                print("INFO: Start date after end date, swapping them")
                date_from_obj, date_to_obj = date_to_obj, date_from_obj
                date_from, date_to = date_to, date_from

            max_lookback = today - timedelta(days=365*5)
            if date_from_obj < max_lookback:
                print("INFO: Start date too far back, limiting to 5 years ago")
                date_from_obj = max_lookback
                date_from = date_from_obj.strftime("%Y-%m-%d")

            print(f"Using dates: {date_from} to {date_to}")
        except ValueError as ve:
            print(f"ERROR: Invalid date format: {ve}")
            return None

        # Download data safely
        def fetch_data(symbol):
            try:
                data = yf.download(symbol, start=date_from, end=date_to, progress=False, threads=False)
                if data.empty and symbol.endswith('.NS'):
                    alt_symbol = symbol.replace('.NS', '')
                    print(f"Retrying with symbol: {alt_symbol}")
                    data = yf.download(alt_symbol, start=date_from, end=date_to, progress=False, threads=False)
                return data
            except Exception as e:
                print(f"ERROR downloading {symbol}: {e}")
                return None

        original_symbol = stock_symbol
        if '.' not in stock_symbol:
            stock_symbol = f"{stock_symbol}.NS"

        data = fetch_data(stock_symbol)
        if data is None or data.empty:
            data = fetch_data(original_symbol)
        if data is None or data.empty:
            print(f"ERROR: No data available for {original_symbol}")
            return None

        print(f"Data downloaded successfully: {len(data)} rows")

        # Extract Close prices safely
        if 'Close' in data.columns:
            prices = data['Close'].dropna().to_numpy(dtype=np.float64)
        else:
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                prices = data[numeric_cols[0]].dropna().to_numpy(dtype=np.float64)
            else:
                print(f"ERROR: Could not find Close prices for {original_symbol}")
                return None

        if len(prices) == 0:
            print(f"ERROR: No price data extracted for {original_symbol}")
            return None

        print(f"Price data extracted: {len(prices)} points, range {prices.min():.2f} to {prices.max():.2f}")

        # Technical indicators
        vol = calculate_volatility(prices)
        sma = calculate_sma(prices)
        ema = calculate_ema(prices, alpha=0.1)
        rsi = calculate_rsi(prices)
        supports, resistances = find_support_resistance(prices)
        support_level = supports.min() if len(supports) > 0 else prices.min()
        resistance_level = resistances.max() if len(resistances) > 0 else prices.max()

        # Signals
        trend = "bullish" if ema > 1.01 * sma else ("bearish" if ema < 0.99 * sma else "neutral")
        momentum_status = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "neutral")
        risk_ratio = vol / sma if sma > 0 else 0
        risk = "High Risk" if risk_ratio > 0.08 else ("Medium Risk" if risk_ratio > 0.04 else "Low Risk")
        final_signal = "BUY" if trend == "bullish" and rsi < 60 and risk != "High Risk" else \
                      ("SELL" if trend == "bearish" and rsi > 40 else "HOLD")

        # AI explanation
        ai_explanation = None
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                current_price_safe = prices[-1].item() if hasattr(prices[-1], "item") else float(prices[-1])
                explanation_prompt = f"""Analyze this stock and provide a brief explanation:
Stock Analysis Summary:
- Current Price: ₹{round(current_price_safe,2)}
- EMA: ₹{round(float(ema),2)}
- SMA: ₹{round(float(sma),2)}
- RSI: {round(float(rsi),2)}
- Volatility: {round((float(vol)/float(sma)*100) if sma>0 else 0,2)}%
- Support: ₹{round(float(support_level),2)}
- Resistance: ₹{round(float(resistance_level),2)}
- Trend: {trend}
- Momentum: {momentum_status}
- Risk: {risk}
- Signal: {final_signal}"""
                
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(explanation_prompt)
                ai_explanation = getattr(response, 'text', str(response))
            except Exception as e:
                print(f"Non-critical AI explanation error: {e}")
                ai_explanation = None

        # Prepare OHLC chart data robustly
        window = min(90, len(prices))
        chart_dates = [d.strftime("%Y-%m-%d") for d in data.index[-window:]]
        chart_ohlc = []
        for i in range(-window, 0):
            o = data['Open'].iloc[i].item() if 'Open' in data.columns else prices[i].item()
            h = data['High'].iloc[i].item() if 'High' in data.columns else prices[i].item()
            l = data['Low'].iloc[i].item() if 'Low' in data.columns else prices[i].item()
            c = data['Close'].iloc[i].item() if 'Close' in data.columns else prices[i].item()
            chart_ohlc.append({
                "x": chart_dates[i + window],
                "o": round(o,2),
                "h": round(h,2),
                "l": round(l,2),
                "c": round(c,2)
            })

        return {
            "volatility": round(float(vol), 2),
            "volatility_percent": round((float(vol)/float(sma)*100) if sma>0 else 0, 2),
            "sma": round(float(sma),2),
            "ema": round(float(ema),2),
            "rsi": round(float(rsi),2),
            "support": round(float(support_level),2),
            "resistance": round(float(resistance_level),2),
            "trend": trend,
            "momentum": momentum_status,
            "risk": risk,
            "signal": final_signal,
            "current_price": round(current_price_safe,2),
            "ai_explanation": ai_explanation,
            "chart": {
                "dates": chart_dates,
                "ohlc": chart_ohlc
            }
        }

    except Exception as e:
        print(f"CRITICAL ERROR analyzing stock {stock_symbol}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "firebase": FIREBASE_AVAILABLE and db is not None,
        "gemini": GEMINI_AVAILABLE and GEMINI_API_KEY is not None
    }), 200

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/analysis')
def analysis():
    return render_template('result.html')

@app.route('/add_expense', methods=['POST'])
def add_expense():
    stock_name = request.form.get('stockName', '').strip()
    date_from_str = request.form.get("dateFrom", '').strip()
    date_to_str = request.form.get("dateTo", '').strip()
    if not stock_name or not date_from_str or not date_to_str:
        return render_template('index.html', error="Please fill in all fields"), 200
    from urllib.parse import quote
    redirect_url = f"/analysis?stock={quote(stock_name)}&date_from={quote(date_from_str)}&date_to={quote(date_to_str)}"
    return redirect(redirect_url, code=302)

@app.route('/api/analyze', methods=['GET', 'POST'])
def api_analyze():
    try:
        if request.method == 'POST':
            data = request.get_json() or {}
            stock_symbol = data.get('stock') or request.form.get('stock')
            date_from = data.get('date_from') or request.form.get('dateFrom')
            date_to = data.get('date_to') or request.form.get('dateTo')
        else:
            stock_symbol = request.args.get('stock')
            date_from = request.args.get('date_from')
            date_to = request.args.get('date_to')
        
        if not stock_symbol or not date_from or not date_to:
            return jsonify({"error": "Missing required parameters: stock, date_from, and date_to are required"}), 400
        
        print(f"=== API Analyze Request ===")
        print(f"Stock: {stock_symbol}, From: {date_from}, To: {date_to}")
        
        original_symbol = stock_symbol
        if '.' not in stock_symbol:
            stock_symbol = f"{stock_symbol}.NS"
            print(f"Converted {original_symbol} to {stock_symbol} (Indian stock)")
        
        result = analyze_stock(stock_symbol, date_from, date_to)
        if result is None and stock_symbol.endswith('.NS') and original_symbol != stock_symbol:
            print(f"Retrying with original symbol: {original_symbol}")
            result = analyze_stock(original_symbol, date_from, date_to)
        
        if result is None:
            return jsonify({"error": f"Failed to analyze stock '{original_symbol}'"}), 400
        
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/api/chat', methods=['POST'])
def api_chat():
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return jsonify({"error": "AI chat is not available. Gemini AI is not configured."}), 503
    
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(message)
        response_text = response.text
        if not response_text:
            return jsonify({"error": "Empty response from AI model"}), 500
        return jsonify({"response": response_text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
