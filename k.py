import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import vnstock as vs
import requests
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from vnstock import Vnstock
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Advanced Stock Analysis Dashboard")

# --- Custom CSS ---
st.markdown("""
<style>
    /* Main background and container */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #333;
    }
    .main .block-container {
        padding: 2rem; max-width: 1400px; margin: 0 auto;
        background: rgba(255, 255, 255, 0.95); border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    /* Header */
    .css-1cpxqw2.e16z0g0x1 {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white;
        padding: 30px; text-align: center; border-radius: 20px 20px 0 0;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; background: #f8f9fa; border-bottom: 2px solid #dee2e6;
        border-radius: 10px 10px 0 0; margin: 0 30px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        padding: 20px; cursor: pointer; transition: all 0.3s; font-weight: 600;
        border-bottom: 3px solid transparent; color: #495057; min-width: 120px;
    }
    .stTabs [data-baseweb="tab-list"] button:hover { background: #e9ecef; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: white; border-bottom-color: #667eea; color: #667eea;
    }
    /* Buttons */
    .stButton>button {
        padding: 15px 25px; border: none; border-radius: 10px; font-size: 16px;
        cursor: pointer; transition: all 0.3s; font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 100%;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
    /* Text Input */
    .stTextInput>div>div>input {
        padding: 15px; border: 2px solid #ddd; border-radius: 10px; font-size: 16px; width: 100%;
    }
    .stTextInput>div>div>input:focus { border-color: #667eea; box-shadow: 0 0 10px rgba(102, 126, 234, 0.3); }
    /* Radio Buttons */
    .stRadio > label { font-weight: 600; margin-bottom: 10px; }
    .stRadio > div { display: flex; flex-wrap: wrap; gap: 10px; margin-left: 15px; }
    .stRadio > div > label {
        padding: 8px 16px; border: 2px solid #667eea; background: white;
        color: #667eea; border-radius: 5px; cursor: pointer; transition: all 0.3s;
        font-size: 14px; font-weight: 500;
    }
    .stRadio > div > label[data-baseweb="radio"] span:first-child { display: none; }
    .stRadio > div > label[data-baseweb="radio"][aria-checked="true"] {
        background: #667eea; color: white; box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    /* Cards */
    .st-emotion-cache-1r6dm1x {
        background: rgba(255, 255, 255, 0.95); border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1); padding: 25px; margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    .st-emotion-cache-1r6dm1x h3 {
        color: #2c3e50; margin-bottom: 15px; font-size: 1.4em;
        border-bottom: 1px solid #eee; padding-bottom: 10px;
    }
    /* Indicator Signals */
    .indicator-signal {
        font-size: 0.9em; padding: 5px 10px; border-radius: 5px; display: inline-block;
        margin-top: 5px; font-weight: 600;
    }
    .signal-buy { background: #d4edda; color: #155724; }
    .signal-sell { background: #f8d7da; color: #721c24; }
    .signal-hold { background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

# --- Custom Header ---
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; text-align: center; border-radius: 20px 20px 0 0;">
    <h1> BẢNG THÔNG TIN PHÂN TÍCH CỔ PHIẾU </h1>
    <p><b>DAP391m Stock Market Analysis</b></p>
</div>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, time_range):
    """Fetch historical stock data and financial metrics."""
    symbol = symbol.strip().upper()
    data = pd.DataFrame()
    is_vietnamese_stock = False
    vn_stock_symbols = ['FPT', 'VCB', 'HPG', 'VIC', 'VND', 'SSI', 'GAS', 'MWG', 'PNJ', 'CTG', 'BID', 'MBB','TPB']

    # Validate symbol format
    if not re.match(r'^[A-Z0-9.-]+$', symbol):
        st.error("Mã cổ phiếu không hợp lệ. Vui lòng nhập mã hợp lệ (VD: NVDA, FPT.VN).")
        return None

    # Vietnamese stock handling
    if symbol in vn_stock_symbols or symbol.endswith('.VN'):
        is_vietnamese_stock = True
        end_date = datetime.now()
        days_map = {'1D': 5, '1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365, '3Y': 3 * 365}
        start_date = end_date - timedelta(days=days_map.get(time_range, 365))

        try:
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=symbol, source='TCBS')
            data_vn = stock.quote.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1D'
            )
            if not data_vn.empty:
                data = data_vn.rename(columns={'time': 'Date'}).set_index('Date')
                data.index = pd.to_datetime(data.index)
            else:
                st.error(f"Không có dữ liệu từ vnstock cho {symbol}. Vui lòng kiểm tra mã hoặc thời gian.")
                return None
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu từ vnstock cho {symbol}: {e}")
            st.info("Vui lòng đảm bảo bạn đã cài đặt phiên bản mới nhất của vnstock: pip install vnstock --upgrade")
            return None
    else:  # International stocks via yfinance
        period_map = {'1D': '5d', '1W': '1mo', '1M': '3mo', '3M': '6mo', '6M': '1y', '1Y': '2y', '3Y': '5y'}
        ticker = yf.Ticker(symbol)
        try:
            data = ticker.history(period=period_map.get(time_range, '3M'), auto_adjust=True)
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu từ yfinance cho {symbol}: {e}")
            return None

    if data.empty:
        st.error(f"Không có dữ liệu cho mã cổ phiếu {symbol}. Vui lòng kiểm tra lại mã.")
        return None

    # Normalize column names
    column_mapping = {
        'open': 'Open', 'Open': 'Open', 'OPEN': 'Open',
        'high': 'High', 'High': 'High', 'HIGH': 'High',
        'low': 'Low', 'Low': 'Low', 'LOW': 'Low',
        'close': 'Close', 'Close': 'Close', 'CLOSE': 'Close',
        'volume': 'Volume', 'Volume': 'Volume', 'VOLUME': 'Volume'
    }
    try:
        data = data.rename(columns=column_mapping)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Dữ liệu thiếu các cột cần thiết: {missing_columns}. Vui lòng kiểm tra nguồn dữ liệu.")
            return None
        data = data[required_columns]
    except Exception as e:
        st.error(f"Lỗi khi chuẩn hóa cột dữ liệu: {e}")
        return None

    data.index = pd.to_datetime(data.index.date)  # Remove timezone

    # Fetch financial metrics
    current_price = data['Close'].iloc[-1]
    previous_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
    change = current_price - previous_close
    change_percent = (change / previous_close) * 100 if previous_close != 0 else 0

    market_cap, pe_ratio, high_52w, low_52w, beta = None, None, None, None, None
    try:
        if is_vietnamese_stock:
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=symbol, source='TCBS')
            overview = stock.company.overview()
            market_cap = overview['market_cap'].iloc[0] * 1_000_000 if 'market_cap' in overview else None
            pe_ratio = overview['pe'].iloc[0] if 'pe' in overview else None
            high_52w = overview['price_high_52w'].iloc[0] if 'price_high_52w' in overview else None
            low_52w = overview['price_low_52w'].iloc[0] if 'price_low_52w' in overview else None
            beta = overview['beta'].iloc[0] if 'beta' in overview else None
        else:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            market_cap = info.get('marketCap')
            pe_ratio = info.get('trailingPE')
            high_52w = info.get('fiftyTwoWeekHigh')
            low_52w = info.get('fiftyTwoWeekLow')
            beta = info.get('beta')
    except Exception:
        st.warning(f"Không thể tải dữ liệu tài chính chi tiết cho {symbol}. Một số giá trị có thể bị thiếu.")

    return {
        'symbol': symbol, 'is_vietnamese': is_vietnamese_stock,
        'currentPrice': current_price, 'change': change, 'changePercent': change_percent,
        'volume': data['Volume'].iloc[-1], 'marketCap': market_cap, 'peRatio': pe_ratio,
        'high52w': high_52w, 'low52w': low_52w, 'beta': beta, 'df': data
    }
def create_indicator_card(title, value, signal, signal_class):
    """Create a styled card for technical indicators."""
    return f"""
    <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); margin-bottom: 10px; height: 100%;">
        <div style="font-weight: bold; color: #2c3e50;">{title}</div>
        <div style="font-size: 1.2em; margin: 5px 0; color: #34495e;">{value}</div>
        <div class="indicator-signal {signal_class}">{signal}</div>
    </div>
    """

def update_stock_info(stock_data):
    """Display stock overview information."""
    st.markdown("<h3>Thông tin cổ phiếu</h3>", unsafe_allow_html=True)
    if not stock_data:
        st.info("Nhấn 'Phân tích' để xem thông tin cổ phiếu.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h4>{stock_data['symbol']}</h4>", unsafe_allow_html=True)
        is_vietnamese = stock_data['is_vietnamese']
        currency_symbol = ' nghìn đồng' if is_vietnamese else '$'
        current_price = f"{abs(stock_data['currentPrice'])}{currency_symbol}" if is_vietnamese else f"${stock_data['currentPrice']:.2f}"
        st.write(f"**Giá hiện tại:** {current_price}")
        change_color = "green" if stock_data['change'] >= 0 else "red"
        change_symbol = "+" if stock_data['change'] >= 0 else ""
        change_value = f"{abs(stock_data['change']):,.0f}{currency_symbol}" if is_vietnamese else f"${stock_data['change']:.2f}"
        st.markdown(f"**Thay đổi:** <span style='color:{change_color}'>{change_symbol}{change_value} ({stock_data['changePercent']:.2f}%)</span>", unsafe_allow_html=True)

    with col2:
        st.write(f"**Khối lượng:** {stock_data['volume']:,}")
        market_cap = stock_data.get('marketCap')
        market_cap_str = f"{market_cap / 1e9:,.0f} tỷ {currency_symbol}" if is_vietnamese and market_cap else f"${market_cap / 1e9:.2f}B" if market_cap else "N/A"
        pe_ratio_str = f"{stock_data['peRatio']:.2f}" if stock_data.get('peRatio') else "N/A"
        st.write(f"**Vốn hóa:** {market_cap_str}")
        st.write(f"**P/E Ratio:** {pe_ratio_str}")

    st.markdown("<h3>Các chỉ số chính</h3>", unsafe_allow_html=True)
    stats_cols = st.columns(4)
    stats = [
        ("Giá hiện tại", current_price),
        ("Thay đổi (%)", f"{change_symbol}{stock_data['changePercent']:.2f}%"),
        ("Khối lượng", f"{stock_data['volume'] / 1e6:.2f}M"),
        ("Vốn hóa thị trường", market_cap_str),
        ("P/E Ratio", pe_ratio_str),
        ("Cao nhất 52W", f"{abs(stock_data['high52w']):,.0f}{currency_symbol}" if is_vietnamese and stock_data.get('high52w') else f"${stock_data['high52w']:.2f}" if stock_data.get('high52w') else "N/A"),
        ("Thấp nhất 52W", f"{abs(stock_data['low52w']):,.0f}{currency_symbol}" if is_vietnamese and stock_data.get('low52w') else f"${stock_data['low52w']:.2f}" if stock_data.get('low52w') else "N/A"),
        ("Beta", f"{stock_data['beta']:.2f}" if stock_data.get('beta') else "N/A")
    ]
    for i, (label, value) in enumerate(stats):
        with stats_cols[i % 4]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #dee2e6; height: 100%;">
                <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{value}</div>
                <div style="color: #6c757d; margin-top: 5px; font-size: 0.9em;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def compute_technical_indicators(df):
    """Compute technical indicators for charts and signals."""
    df = df.copy()
    df['MA5'] = ta.sma(df['Close'], length=5)
    df['MA20'] = ta.sma(df['Close'], length=20)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        df = df.join(macd_df)
    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df = df.join(bbands)
    stoch_df = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
    if stoch_df is not None and not stoch_df.empty:
        df = df.join(stoch_df)
    df['WPR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx_df is not None and not adx_df.empty:
        df = df.join(adx_df)
    return df

def update_charts(stock_data):
    """Display price, volume, candlestick, and technical indicator charts."""
    st.markdown("<h3>Biểu đồ giá và chỉ báo</h3>", unsafe_allow_html=True)
    if not stock_data or stock_data['df'].empty:
        st.info("Không có dữ liệu biểu đồ để hiển thị.")
        return

    df = compute_technical_indicators(stock_data['df'])
    dates = df.index

    # Price Chart with MAs
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=dates, y=df['Close'], mode='lines', name='Giá đóng cửa', line=dict(color='#2c3e50')))
    fig_price.add_trace(go.Scatter(x=dates, y=df['MA5'], mode='lines', name='MA 5', line=dict(color='#28a745', dash='dash')))
    fig_price.add_trace(go.Scatter(x=dates, y=df['MA20'], mode='lines', name='MA 20', line=dict(color='#ff9f40', dash='dash')))
    fig_price.update_layout(title='Biểu đồ giá và đường trung bình', xaxis_title='Ngày', yaxis_title='Giá', xaxis_rangeslider_visible=False, height=400, template="plotly_white")
    st.plotly_chart(fig_price, use_container_width=True)

    # Volume Chart
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=dates, y=df['Volume'], name='Khối lượng', marker_color='#764ba2'))
    fig_volume.update_layout(title='Biểu đồ khối lượng giao dịch', xaxis_title='Ngày', yaxis_title='Khối lượng', height=300, template="plotly_white")
    st.plotly_chart(fig_volume, use_container_width=True)

    # Candlestick Chart
    fig_candle = go.Figure(data=[go.Candlestick(x=dates, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], increasing_line_color='#28a745', decreasing_line_color='#dc3545')])
    fig_candle.update_layout(title='Biểu đồ nến', xaxis_title='Ngày', yaxis_title='Giá', xaxis_rangeslider_visible=False, height=400, template="plotly_white")
    st.plotly_chart(fig_candle, use_container_width=True)

    # RSI Chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=dates, y=df['RSI'], mode='lines', name='RSI (14)', line=dict(color='#007bff')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color='#dc3545', name='Quá mua (70)')
    fig_rsi.add_hline(y=30, line_dash="dash", line_color='#28a745', name='Quá bán (30)')
    fig_rsi.update_layout(title='Biểu đồ RSI', yaxis_range=[0, 100], height=300, template="plotly_white")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD Chart
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=dates, y=df.get('MACD_12_26_9'), mode='lines', name='MACD Line', line=dict(color='#007bff')))
    fig_macd.add_trace(go.Scatter(x=dates, y=df.get('MACDs_12_26_9'), mode='lines', name='Signal Line', line=dict(color='#ffc107', dash='dash')))
    fig_macd.add_trace(go.Bar(x=dates, y=df.get('MACDh_12_26_9'), name='Histogram', marker_color=['#28a745' if val >= 0 else '#dc3545' for val in df.get('MACDh_12_26_9', [])]))
    fig_macd.update_layout(title='Biểu đồ MACD', height=300, template="plotly_white")
    st.plotly_chart(fig_macd, use_container_width=True)

    # Bollinger Bands
    fig_bbands = go.Figure()
    fig_bbands.add_trace(go.Scatter(x=dates, y=df['Close'], mode='lines', name='Giá đóng cửa', line=dict(color='#2c3e50')))
    fig_bbands.add_trace(go.Scatter(x=dates, y=df.get('BBU_20_2.0'), mode='lines', name='Upper Band', line=dict(color='#ff7f0e', dash='dash')))
    fig_bbands.add_trace(go.Scatter(x=dates, y=df.get('BBM_20_2.0'), mode='lines', name='Middle Band', line=dict(color='#1f77b4', dash='dash')))
    fig_bbands.add_trace(go.Scatter(x=dates, y=df.get('BBL_20_2.0'), mode='lines', name='Lower Band', line=dict(color='#2ca02c', dash='dash')))
    fig_bbands.update_layout(title='Biểu đồ Bollinger Bands', height=400, template="plotly_white")
    st.plotly_chart(fig_bbands, use_container_width=True)

def update_technical_indicators(stock_data):
    """Display technical indicators with buy/sell/hold signals."""
    st.markdown("<h3>Các chỉ báo kỹ thuật</h3>", unsafe_allow_html=True)
    if not stock_data or stock_data['df'].empty:
        st.info("Không có dữ liệu để hiển thị.")
        return

    df = stock_data['df'].copy()
    prices = df['Close']

    # Moving Averages
    st.markdown("<h4>Moving Averages (MA)</h4>", unsafe_allow_html=True)
    ma_periods = [5, 10, 20, 50, 100, 200]
    ma_cols = st.columns(3)
    for i, period in enumerate(ma_periods):
        ma_values = ta.sma(prices, length=period)
        if ma_values is not None and not ma_values.empty:
            latest_ma = ma_values.iloc[-1]
            signal = "Giữ"
            signal_class = "signal-hold"
            if not np.isnan(latest_ma):
                if prices.iloc[-1] > latest_ma: signal, signal_class = "Tín hiệu: Mua", "signal-buy"
                elif prices.iloc[-1] < latest_ma: signal, signal_class = "Tín hiệu: Bán", "signal-sell"
            ma_cols[i % 3].markdown(create_indicator_card(f"MA {period}", f"{latest_ma:.2f}", signal, signal_class), unsafe_allow_html=True)

    # Exponential Moving Averages
    st.markdown("<h4>Exponential Moving Averages (EMA)</h4>", unsafe_allow_html=True)
    ema_periods = [12, 26, 50, 100]
    ema_cols = st.columns(4)
    for i, period in enumerate(ema_periods):
        ema_values = ta.ema(prices, length=period)
        if ema_values is not None and not ema_values.empty:
            latest_ema = ema_values.iloc[-1]
            signal, signal_class = "Giữ", "signal-hold"
            if not np.isnan(latest_ema):
                if prices.iloc[-1] > latest_ema: signal, signal_class = "Tín hiệu: Mua", "signal-buy"
                elif prices.iloc[-1] < latest_ema: signal, signal_class = "Tín hiệu: Bán", "signal-sell"
            ema_cols[i % 4].markdown(create_indicator_card(f"EMA {period}", f"{latest_ema:.2f}", signal, signal_class), unsafe_allow_html=True)

    # Other Indicators
    st.markdown("<h4>Các chỉ báo khác</h4>", unsafe_allow_html=True)
    other_cols = st.columns(3)
    
    # RSI
    rsi_series = ta.rsi(df['Close'], length=14)
    if rsi_series is not None and not rsi_series.empty:
        latest_rsi = rsi_series.iloc[-1]
        rsi_signal, rsi_signal_class = "Trung lập", "signal-hold"
        if not np.isnan(latest_rsi):
            if latest_rsi > 70: rsi_signal, rsi_signal_class = "Quá mua", "signal-sell"
            elif latest_rsi < 30: rsi_signal, rsi_signal_class = "Quá bán", "signal-buy"
        other_cols[0].markdown(create_indicator_card("RSI (14)", f"{latest_rsi:.2f}", rsi_signal, rsi_signal_class), unsafe_allow_html=True)

    # MACD
    macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        latest_macd = macd_df['MACD_12_26_9'].iloc[-1]
        latest_signal = macd_df['MACDs_12_26_9'].iloc[-1]
        macd_signal, macd_class = "Bán", "signal-sell"
        if not np.isnan(latest_macd) and not np.isnan(latest_signal) and latest_macd > latest_signal:
            macd_signal, macd_class = "Mua", "signal-buy"
        other_cols[1].markdown(create_indicator_card("MACD (12,26,9)", f"MACD: {latest_macd:.2f}", macd_signal, macd_class), unsafe_allow_html=True)

    # ADX
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx_df is not None and not adx_df.empty and 'ADX_14' in adx_df:
        latest_adx = adx_df['ADX_14'].iloc[-1]
        adx_signal, adx_class = "Xu hướng yếu", "signal-hold"
        if not np.isnan(latest_adx):
            if latest_adx > 25: adx_signal, adx_class = "Xu hướng mạnh", "signal-buy"
        other_cols[2].markdown(create_indicator_card("ADX (14)", f"{latest_adx:.2f}", adx_signal, adx_class), unsafe_allow_html=True)

def update_oscillators(stock_data):
    """Display oscillator indicators (Stochastic, Williams %R)."""
    st.markdown("<h3>Các chỉ báo dao động</h3>", unsafe_allow_html=True)
    if not stock_data or stock_data['df'].empty:
        st.info("Không có dữ liệu để hiển thị.")
        return

    df = compute_technical_indicators(stock_data['df'])
    dates = df.index

    # Stochastic Oscillator
    fig_stoch = go.Figure()
    fig_stoch.add_trace(go.Scatter(x=dates, y=df.get('STOCHk_14_3_3'), mode='lines', name='%K'))
    fig_stoch.add_trace(go.Scatter(x=dates, y=df.get('STOCHd_14_3_3'), mode='lines', name='%D'))
    fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
    fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
    fig_stoch.update_layout(title='Stochastic Oscillator (14,3,3)', height=300, template="plotly_white", yaxis_range=[0,100])
    st.plotly_chart(fig_stoch, use_container_width=True)

    # Williams %R
    fig_wpr = go.Figure()
    fig_wpr.add_trace(go.Scatter(x=dates, y=df.get('WPR'), mode='lines', name='W%R'))
    fig_wpr.add_hline(y=-20, line_dash="dash", line_color="red")
    fig_wpr.add_hline(y=-80, line_dash="dash", line_color="green")
    fig_wpr.update_layout(title='Williams %R (14)', height=300, template="plotly_white", yaxis_range=[-100,0])
    st.plotly_chart(fig_wpr, use_container_width=True)

def update_fundamentals(stock_data):
    """Display fundamental financial data."""
    st.markdown("<h3>Phân tích cơ bản</h3>", unsafe_allow_html=True)
    if not stock_data:
        st.info("Không có dữ liệu.")
        return

    symbol = stock_data['symbol']
    try:
        with st.spinner(f"Đang tải dữ liệu tài chính cho {symbol}..."):
            if stock_data['is_vietnamese']:
                from vnstock import Vnstock
                stock = Vnstock().stock(symbol=symbol, source='TCBS')
                
                # Fetch financial ratios
                st.subheader("Chỉ số tài chính (Theo Quý)")
                df_ratios = stock.finance.ratio(period='quarter', lang='en')  # Use stock.finance.ratio
                st.dataframe(df_ratios)
                
                # Fetch income statement
                st.subheader("Báo cáo kết quả kinh doanh (Theo Quý)")
                df_income = stock.finance.income_statement(period='quarter', lang='en')  # Use stock.finance.income_statement
                st.dataframe(df_income)
            else:
                ticker = yf.Ticker(symbol)
                st.subheader("Báo cáo tài chính (Hàng năm)")
                st.dataframe(ticker.financials)
                
                st.subheader("Bảng cân đối kế toán (Hàng năm)")
                st.dataframe(ticker.balance_sheet)
    except Exception as e:
        st.error(f"Không thể tải dữ liệu tài chính: {e}")

def update_comparison(symbols_string, time_range):
    """Compare performance of multiple stocks."""
    st.markdown("<h3>So sánh các cổ phiếu</h3>", unsafe_allow_html=True)
    symbols = [s.strip().upper() for s in symbols_string.split(',') if s.strip()]

    if len(symbols) < 2:
        st.info("Nhập ít nhất 2 mã cổ phiếu (cách nhau bởi dấu phẩy) để so sánh.")
        return

    fig = go.Figure()
    all_data = {}
    with st.spinner("Đang tải dữ liệu so sánh..."):
        for symbol in symbols:
            data = fetch_stock_data(symbol, time_range)
            if data and not data['df'].empty:
                all_data[symbol] = data['df']

    if not all_data:
        st.warning("Không tải được dữ liệu cho bất kỳ mã nào được nhập.")
        return

    for symbol, df in all_data.items():
        # normalized_price = (df['Close'] / df['Close'].iloc[0] - 1) * 100
        normalized_price = df['Close']
        fig.add_trace(go.Scatter(x=normalized_price.index, y=normalized_price, mode='lines', name=symbol))

    fig.update_layout(
        title='So sánh giá ',
        xaxis_title='Ngày', yaxis_title='Giá',
        template="plotly_white", height=500
    )
    st.plotly_chart(fig, use_container_width=True)
def update_comparison_volume(symbols_string, time_range):
    """Compare performance of multiple stocks."""
    st.markdown("<h3>So sánh các cổ phiếu</h3>", unsafe_allow_html=True)
    symbols = [s.strip().upper() for s in symbols_string.split(',') if s.strip()]

    if len(symbols) < 2:
        st.info("Nhập ít nhất 2 mã cổ phiếu (cách nhau bởi dấu phẩy) để so sánh.")
        return

    fig = go.Figure()
    all_data = {}
    with st.spinner("Đang tải dữ liệu so sánh..."):
        for symbol in symbols:
            data = fetch_stock_data(symbol, time_range)
            if data and not data['df'].empty:
                all_data[symbol] = data['df']

    if not all_data:
        st.warning("Không tải được dữ liệu cho bất kỳ mã nào được nhập.")
        return

    for symbol, df in all_data.items():
        # normalized_price = (df['Close'] / df['Close'].iloc[0] - 1) * 100
        normalized_price = df['Volume']
        fig.add_trace(go.Scatter(x=normalized_price.index, y=normalized_price, mode='lines', name=symbol))

    fig.update_layout(
        title='So sánh khối lượng giao dịch ',
        xaxis_title='Ngày', yaxis_title='Volume',
        template="plotly_white", height=500
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def train_arima_model(df_close):
    """Train an Auto-ARIMA model and return predictions."""
    model = pm.auto_arima(df_close, seasonal=False, stepwise=True, trace=False)
    forecast, conf_int = model.predict(n_periods=30, return_conf_int=True)
    return forecast, conf_int

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

from prophet import Prophet
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def update_predictions(stock_data):
    """Display Prophet-based price predictions with enhanced visualization."""
    st.markdown("<h3>Dự đoán giá  </h3>", unsafe_allow_html=True)
    if not stock_data or stock_data['df'].empty or len(stock_data['df']) < 30:
        st.info("Cần ít nhất 30 ngày dữ liệu để tạo dự đoán.")
        return

    df_close = stock_data['df']['Close'].reset_index()
    df_close.columns = ['ds', 'y']  # Prophet requires 'ds' for date and 'y' for value
    is_vietnamese = stock_data['is_vietnamese']
    currency_symbol = ' nghìn đồng' if is_vietnamese else '$'

    try:
        with st.spinner("Đang huấn luyện mô hình dự đoán..."):
            # Train Prophet model
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, interval_width=0.95)
            model.fit(df_close)

            # Predict for the next 14 days
            future = model.make_future_dataframe(periods=14)
            forecast = model.predict(future)

            # Extract predictions for 1, 7, and 14 days
            pred1day = forecast.iloc[-14]['yhat']  # Day 1 of forecast
            pred7day = forecast.iloc[-7]['yhat']   # Day 7 of forecast
            pred14day = forecast.iloc[-1]['yhat']  # Day 14 of forecast

            # Display predictions
            pred_cols = st.columns(3)
            predictions = [("1 ngày", pred1day), ("7 ngày", pred7day)]
            for i, (label, value) in enumerate(predictions):
                formatted_value = f"{abs(value):,.2f}{currency_symbol}" if is_vietnamese else f"${value:.2f}"
                with pred_cols[i]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4); height: 100%;">
                        <h4>Dự đoán {label}</h4>
                        <div style="font-size: 2.5em; font-weight: bold; margin: 10px 0;">{formatted_value}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Plot forecast with enhanced visualization
            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                x=df_close['ds'][-30:], y=df_close['y'][-30:],
                mode='lines', name='Lịch sử',
                line=dict(color='#2c3e50', width=2)
            ))

            # Forecast data with markers for key points
            forecast_dates = forecast['ds'][-14:]
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast['yhat'][-14:],
                mode='lines+markers', name='Dự đoán',
                line=dict(color='#667eea', width=2),
                marker=dict(size=8, color='#667eea')
            ))
            fig.add_trace(go.Scatter(
                x=[forecast_dates[0]], y=[pred1day],
                mode='markers+text', name='Ngày 1',
                text=[f"{abs(pred1day):,.0f}{currency_symbol}" if is_vietnamese else f"${pred1day:.2f}"],
                textposition="top center",
                marker=dict(size=10, color='#ff7f0e'),
                textfont=dict(size=12)
            ))
            fig.add_trace(go.Scatter(
                x=[forecast_dates[6]], y=[pred7day],
                mode='markers+text', name='Ngày 7',
                text=[f"{abs(pred7day):,.0f}{currency_symbol}" if is_vietnamese else f"${pred7day:.2f}"],
                textposition="top center",
                marker=dict(size=10, color='#ff7f0e'),
                textfont=dict(size=12)
            ))
            fig.add_trace(go.Scatter(
                x=[forecast_dates[13]], y=[pred14day],
                mode='markers+text', name='Ngày 14',
                text=[f"{abs(pred14day):,.0f}{currency_symbol}" if is_vietnamese else f"${pred14day:.2f}"],
                textposition="top center",
                marker=dict(size=10, color='#ff7f0e'),
                textfont=dict(size=12)
            ))

            # Confidence Interval as filled area
            fig.add_trace(go.Scatter(
                x=forecast_dates.tolist() + forecast_dates[::-1].tolist(),
                y=forecast['yhat_upper'][-14:].tolist() + forecast['yhat_lower'][-14:][::-1].tolist(),
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='Khoảng tin cậy',
                showlegend=True
            ))

            fig.update_layout(
                title='Dự đoán giá 14 ngày (Prophet)',
                xaxis_title='Ngày',
                yaxis_title=f'Giá ({currency_symbol})',
                template="plotly_white",
                height=400,
                yaxis=dict(range=[min(df_close['y'][-30:].min(), forecast['yhat_lower'].min()) - 5, 
                                max(df_close['y'][-30:].max(), forecast['yhat_upper'].max()) + 5]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<p style='font-size: 0.9em; color: #6c757d; margin-top: 15px;'><em>Lưu ý: Đây là dự đoán từ mô hình toán học, không phải lời khuyên tài chính.</em></p>", unsafe_allow_html=True)

    except Exception as e:
        # st.error(f"Lỗi khi xây dựng mô hình dự đoán: {e}")
        pass


def update_news(symbol):
    """Fetch and display recent news for the stock."""
    st.markdown("<h3>Tin tức và Phân tích</h3>", unsafe_allow_html=True)
    try:
        api_key = st.secrets["finnhub"]["API_KEY"]
    except KeyError:
        st.error("API key không được cấu hình. Vui lòng thêm Finnhub API key vào .streamlit/secrets.toml.")
        return

    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date}&to={end_date}&token={api_key}'
        with st.spinner("Đang tải tin tức..."):
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            news_list = r.json()

        if not news_list:
            st.info(f"Không tìm thấy tin tức gần đây cho mã {symbol}.")
            return
        analyzer = SentimentIntensityAnalyzer()


        for news_item in news_list[:5]:
            news_date = datetime.fromtimestamp(news_item['datetime']).strftime('%d-%m-%Y')
            headline = news_item['headline']
            summary = news_item['summary']
            text = headline + ' ' + summary
            sentiment_score = analyzer.polarity_scores(text)['compound']
            if sentiment_score > 0.5 :
                sentiment = 'tich cuc'
                sentiment_color = 'green'
            else :
                sentiment = 'tieu cuc'
                sentiment_color = 'red'
        
            

            st.markdown(f"""
            <div style="background: white; border-radius: 15px; padding: 20px; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1); border: 1px solid #e9ecef; margin-bottom: 20px;">
                <div style="font-weight: bold; color: #2c3e50; margin-bottom: 10px; font-size: 1.1em;">
                    <a href="{news_item['url']}" target="_blank" style="text-decoration: none; color: inherit;">{news_item['headline']}</a>
                </div>
                <div style="color: #6c757d; font-size: 0.9em; line-height: 1.4;">{news_item['summary']}</div>
                <div style="color: #adb5bd; font-size: 0.8em; margin-top: 10px; text-align: right;">{news_date} - Nguồn: {news_item['source']}</div>
                <div style="margin-top : 8px; font-weight :bold ,color {sentiment_color};">
                danh gia {sentiment}</div>
            </div>
            """, unsafe_allow_html=True)

    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối đến API tin tức: {e}")
    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu tin tức: {e}")

# --- Main Application Layout ---
col1, col2, col3 = st.columns([3, 4, 1])
with col1:
    symbol_input = st.text_input("Nhập mã cổ phiếu (VD: NVDA hoặc FPT,VCB,HPG)", "NVDA").upper()
with col2:
    time_range_options = ['1D', '1W', '1M', '3M', '6M', '1Y', '3Y']
    if 'selected_time_range' not in st.session_state:
        st.session_state.selected_time_range = '3M'
    selected_time_range = st.radio(
        "Chọn phạm vi thời gian:", time_range_options,
        index=time_range_options.index(st.session_state.selected_time_range),
        horizontal=True, key='time_range_radio'
    )
    st.session_state.selected_time_range = selected_time_range
with col3:
    st.write("")
    st.write("")
    analyze_button = st.button("Phân tích")

# Process first symbol for main analysis
main_symbol = [s.strip() for s in symbol_input.split(',') if s.strip()][0]
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

if analyze_button:
    with st.spinner(f'Đang tải và phân tích {main_symbol}...'):
        st.session_state.stock_data = fetch_stock_data(main_symbol, st.session_state.selected_time_range)

# Create tabs
tab_list = ["Tổng quan", "Biểu đồ", "Chỉ báo kỹ thuật", "Oscillators", "Phân tích cơ bản", "So sánh", "Dự đoán", "Tin tức"]
tabs = st.tabs(tab_list)

with tabs[0]:
    update_stock_info(st.session_state.stock_data)
with tabs[1]:
    update_charts(st.session_state.stock_data)
with tabs[2]:
    update_technical_indicators(st.session_state.stock_data)
with tabs[3]:
    update_oscillators(st.session_state.stock_data)
with tabs[4]:
    update_fundamentals(st.session_state.stock_data)
with tabs[5]:
    update_comparison(symbol_input, st.session_state.selected_time_range)
    update_comparison_volume(symbol_input, st.session_state.selected_time_range)
with tabs[6]:
    update_predictions(st.session_state.stock_data)
    
with tabs[7]:
    update_news(main_symbol)