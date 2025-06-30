import streamlit as st
import pandas as pd
import numpy as np
from data_loader import fetch_stock_data
from preprocessing import preprocess_data
from model import build_lstm_model, train_lstm_model
from predict import make_predictions, inverse_transform, calculate_rmse, plotly_predictions
from future_predict import predict_future
from sentiment_analysis import get_sentiment_analysis
from datetime import datetime, timedelta
import plotly.graph_objs as go

# Top 25 US (S&P 500) stocks
us_stocks = [
    ("Apple Inc. (AAPL)", "AAPL"),
    ("Microsoft Corp. (MSFT)", "MSFT"),
    ("Amazon.com Inc. (AMZN)", "AMZN"),
    ("NVIDIA Corp. (NVDA)", "NVDA"),
    ("Alphabet Inc. (GOOGL)", "GOOGL"),
    ("Meta Platforms (META)", "META"),
    ("Berkshire Hathaway (BRK-B)", "BRK-B"),
    ("Eli Lilly (LLY)", "LLY"),
    ("JPMorgan Chase (JPM)", "JPM"),
    ("Visa Inc. (V)", "V"),
    ("Exxon Mobil (XOM)", "XOM"),
    ("UnitedHealth (UNH)", "UNH"),
    ("Johnson & Johnson (JNJ)", "JNJ"),
    ("Procter & Gamble (PG)", "PG"),
    ("Mastercard (MA)", "MA"),
    ("Broadcom (AVGO)", "AVGO"),
    ("Home Depot (HD)", "HD"),
    ("Chevron (CVX)", "CVX"),
    ("Merck & Co. (MRK)", "MRK"),
    ("Costco (COST)", "COST"),
    ("AbbVie (ABBV)", "ABBV"),
    ("Coca-Cola (KO)", "KO"),
    ("PepsiCo (PEP)", "PEP"),
    ("Walmart (WMT)", "WMT"),
    ("Walt Disney (DIS)", "DIS")
]

# Top 25 Indian (Nifty 50) stocks
indian_stocks = [
    ("Reliance Industries (RELIANCE.NS)", "RELIANCE.NS"),
    ("HDFC Bank (HDFCBANK.NS)", "HDFCBANK.NS"),
    ("ICICI Bank (ICICIBANK.NS)", "ICICIBANK.NS"),
    ("Infosys (INFY.NS)", "INFY.NS"),
    ("TCS (TCS.NS)", "TCS.NS"),
    ("Bharti Airtel (BHARTIARTL.NS)", "BHARTIARTL.NS"),
    ("Larsen & Toubro (LT.NS)", "LT.NS"),
    ("ITC (ITC.NS)", "ITC.NS"),
    ("State Bank of India (SBIN.NS)", "SBIN.NS"),
    ("Hindustan Unilever (HINDUNILVR.NS)", "HINDUNILVR.NS"),
    ("Axis Bank (AXISBANK.NS)", "AXISBANK.NS"),
    ("Kotak Mahindra Bank (KOTAKBANK.NS)", "KOTAKBANK.NS"),
    ("Bajaj Finance (BAJFINANCE.NS)", "BAJFINANCE.NS"),
    ("Maruti Suzuki (MARUTI.NS)", "MARUTI.NS"),
    ("HCL Technologies (HCLTECH.NS)", "HCLTECH.NS"),
    ("Sun Pharma (SUNPHARMA.NS)", "SUNPHARMA.NS"),
    ("Asian Paints (ASIANPAINT.NS)", "ASIANPAINT.NS"),
    ("Titan Company (TITAN.NS)", "TITAN.NS"),
    ("UltraTech Cement (ULTRACEMCO.NS)", "ULTRACEMCO.NS"),
    ("Nestle India (NESTLEIND.NS)", "NESTLEIND.NS"),
    ("Power Grid Corp (POWERGRID.NS)", "POWERGRID.NS"),
    ("NTPC (NTPC.NS)", "NTPC.NS"),
    ("JSW Steel (JSWSTEEL.NS)", "JSWSTEEL.NS"),
    ("Bajaj Finserv (BAJAJFINSV.NS)", "BAJAJFINSV.NS"),
    ("Tata Motors (TATAMOTORS.NS)", "TATAMOTORS.NS")
]

st.set_page_config(page_title='AI-Powered Stock Price Predictor', layout='wide')
st.title('📈 AI-Powered Stock Price Predictor')

st.sidebar.header('Stock Selection')
stock_market = st.sidebar.selectbox('Choose Market', ['Top US Stocks', 'Top Indian Stocks', 'Manual Entry'])

if stock_market == 'Top US Stocks':
    stock_choice = st.sidebar.selectbox('Select Stock', us_stocks, format_func=lambda x: x[0])
    ticker = stock_choice[1]
elif stock_market == 'Top Indian Stocks':
    stock_choice = st.sidebar.selectbox('Select Stock', indian_stocks, format_func=lambda x: x[0])
    ticker = stock_choice[1]
else:
    ticker = st.sidebar.text_input('Enter Stock Ticker', value='AAPL')

st.sidebar.header('Settings')
start_date = st.sidebar.date_input('Start Date', value=datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input('End Date', value=datetime.now())
n_lags = st.sidebar.slider('Lag Days', 1, 10, 3)
add_ma = st.sidebar.checkbox('Add Moving Averages', value=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Historical Prediction", "🔮 Future Prediction", "📰 News Sentiment", "📈 Combined Analysis"])

if st.sidebar.button('Analyze'):
    with st.spinner('Fetching data...'):
        df = fetch_stock_data(ticker, str(start_date), str(end_date))
    
    if df.empty:
        st.error('No data found for this ticker.')
    else:
        with st.spinner('Preprocessing data...'):
            df_scaled, scaler = preprocess_data(df, n_lags=n_lags, add_ma=add_ma)
        
        # Prepare features and targets
        X = df_scaled.drop('Close', axis=1).values
        y = df_scaled['Close'].values
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Split train/test
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        with st.spinner('Training model...'):
            model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
            train_lstm_model(model, X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
        
        # Tab 1: Historical Prediction
        with tab1:
            st.write(f"### Historical Prediction for {ticker}")
            with st.spinner('Making predictions...'):
                y_pred = make_predictions(model, X_test)
                y_pred_inv = inverse_transform(scaler, y_pred)
                y_test_inv = inverse_transform(scaler, y_test)
                rmse = calculate_rmse(y_test_inv, y_pred_inv)
            
            st.write(f"**RMSE: {rmse:.2f}**")
            st.plotly_chart(plotly_predictions(y_test_inv, y_pred_inv, f'{ticker} - Actual vs Predicted'), use_container_width=True)
            # Enhanced detailed table for last 10 predictions
            last_10_idx = df_scaled.index[-len(y_test_inv):][-10:]
            detailed_df = pd.DataFrame({
                'Date': last_10_idx,
                'Actual': y_test_inv[-10:],
                'Predicted': y_pred_inv[-10:]
            })
            detailed_df['Absolute Error'] = (detailed_df['Actual'] - detailed_df['Predicted']).abs()
            detailed_df['% Error'] = (detailed_df['Absolute Error'] / detailed_df['Actual'] * 100).round(2)
            st.write('#### Last 10 Actual vs Predicted (Detailed)')
            st.dataframe(detailed_df)
        
        # Tab 2: Future Prediction
        with tab2:
            st.write(f"### Future Prediction for {ticker}")
            future_days = st.slider('Days to Predict', 1, 30, 7)
            
            with st.spinner('Predicting future prices...'):
                last_data = X[-1]  # Last available data point
                future_dates, future_prices = predict_future(model, scaler, last_data, future_days)
            
            # Create future prediction plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines+markers', name='Predicted'))
            fig.update_layout(title=f'{ticker} - Future Price Prediction', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig, use_container_width=True)
            
            # Display future predictions table
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices})
            st.write('#### Future Price Predictions')
            st.dataframe(future_df)
        
        # Tab 3: News Sentiment
        with tab3:
            st.write(f"### News Sentiment Analysis for {ticker}")
            with st.spinner('Analyzing news sentiment...'):
                sentiment_data = get_sentiment_analysis(ticker)
            
            # Display sentiment score
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentiment", sentiment_data['sentiment'].title())
            with col2:
                st.metric("Sentiment Score", f"{sentiment_data['score']:.3f}")
            with col3:
                st.metric("Headlines Analyzed", len(sentiment_data['headlines']))
            
            # Display headlines with sentiment
            if sentiment_data['headlines']:
                st.write('#### Recent Headlines')
                for i, (headline, sentiment) in enumerate(zip(sentiment_data['headlines'], sentiment_data['sentiments'])):
                    color = "green" if sentiment > 0 else "red" if sentiment < 0 else "gray"
                    st.markdown(f"<span style='color:{color}'>{headline}</span>", unsafe_allow_html=True)
        
        # Tab 4: Combined Analysis
        with tab4:
            st.write(f"### Combined Analysis for {ticker}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Model Performance**")
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"Data Points: {len(df)}")
                st.write(f"Training Period: {start_date} to {end_date}")
            
            with col2:
                st.write("**Sentiment Overview**")
                st.write(f"Overall Sentiment: {sentiment_data['sentiment'].title()}")
                st.write(f"Sentiment Score: {sentiment_data['score']:.3f}")
                st.write(f"News Articles: {len(sentiment_data['headlines'])}")
            
            # Combined recommendation
            st.write("**📋 Analysis Summary**")
            if sentiment_data['score'] > 0.1 and rmse < 10:
                st.success("✅ Positive sentiment with good model accuracy - Consider bullish outlook")
            elif sentiment_data['score'] < -0.1:
                st.warning("⚠️ Negative sentiment detected - Exercise caution")
            else:
                st.info("ℹ️ Neutral sentiment - Monitor for changes") 