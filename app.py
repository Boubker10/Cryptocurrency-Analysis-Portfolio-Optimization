import streamlit as st
import pandas as pd
import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as optimize
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import yfinance as yf
from datetime import datetime, timedelta
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

def plot_time_series(data, title, ylabel, key):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Prix Ajusté', line=dict(color='blue')))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=ylabel, hovermode="x unified")
    st.plotly_chart(fig, key=key)
    return fig


def plot_log_returns(data, key):
    log_returns = np.log(data / data.shift(1)).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=log_returns.index, y=log_returns, mode='lines', name='Log Returns', line=dict(color='green')))
    fig.update_layout(title="Rendement (Log Returns)", xaxis_title='Date', yaxis_title='Log Returns', hovermode="x unified")
    st.plotly_chart(fig, key=key)
    return fig, log_returns


def adf_test(series):
    result = adfuller(series.dropna())
    st.write(f"Test de Dickey-Fuller augmenté :")
    st.write(f"Statistique de test : {result[0]}")
    st.write(f"p-value : {result[1]}")
    st.write(f"Valeurs critiques : {result[4]}")


def plot_acf_pacf(data):
    fig_acf = plt.figure(figsize=(12, 6))
    ax1 = fig_acf.add_subplot(211)
    plot_acf(data, ax=ax1, lags=30)
    ax2 = fig_acf.add_subplot(212)
    plot_pacf(data, ax=ax2, lags=30)
    st.pyplot(fig_acf) 


def load_yfinance_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Date'] = stock_data.index
    stock_data.set_index('Date', inplace=True)
    return stock_data


def run_prophet(stock_data_adj_close, forecast_steps):
    df_prophet = stock_data_adj_close.reset_index().rename(columns={"Date": "ds", "Adj Close": "y"})
    model_prophet = Prophet(daily_seasonality=True)
    model_prophet.fit(df_prophet)
    
    future_dates = model_prophet.make_future_dataframe(periods=forecast_steps)
    forecast = model_prophet.predict(future_dates)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Données Historiques', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prévisions', line=dict(color='red')))
    fig.update_layout(title=f'Prévisions avec Prophet pour {forecast_steps} jours', xaxis_title='Date', yaxis_title='Prix Ajusté (USD)', hovermode="x unified")
    st.plotly_chart(fig, key='prophet_forecast')



st.set_page_config(layout="wide", page_title="Crypto Analysis & Prediction")

START_DATE = "2023-01-01"
RSI_TIME_WINDOW = 7

urls = ["https://www.cryptodatadownload.com/cdd/Bitfinex_EOSUSD_d.csv", 
        "https://www.cryptodatadownload.com/cdd/Bitfinex_EDOUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_BTCUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_ETHUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_LTCUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_BATUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_OMGUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_DAIUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_ETCUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_ETPUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_NEOUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_REPUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_TRXUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_XLMUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_XMRUSD_d.csv",
        "https://www.cryptodatadownload.com/cdd/Bitfinex_XVGUSD_d.csv"]
crypto_names= ["EOS Coin (EOS)",
                "Eidoo (EDO)",
                "Bitcoin (BTC)",
                "Ethereum (ETH)",
                "Litecoin (LTC)",
                "Basic Attention Token (BAT)",
                "OmiseGO (OMG)",
                "Dai (DAI)",
                "Ethereum Classic (ETC)",
                "Metaverse (ETP)",
                "Neo (NEO)",
                "Augur (REP)",
                "TRON (TRX)",
                "Stellar (XLM)",
                "Monero (XMR)",
                "Verge (XVG)"]



def train_prophet_model(df, crypto_name, periods=30):
    """
    Train Facebook Prophet model for cryptocurrency prediction
    
    Args:
        df (pd.DataFrame): DataFrame containing the cryptocurrency data
        crypto_name (str): Name of the cryptocurrency
        periods (int): Number of days to forecast
    
    Returns:
        dict: Contains forecast, model and metrics
    """
    prophet_df = df.reset_index()[['date', 'close']].rename(
        columns={'date': 'ds', 'close': 'y'}
    )
    
    model = Prophet(
        changepoint_prior_scale=0.05,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        interval_width=0.95
    )
    
    with st.spinner(f'Training Prophet model for {crypto_name}...'):
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        y_true = prophet_df['y'].values
        y_pred = forecast['yhat'][:len(y_true)]
        
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'Forecast Period': f'{periods} days'
        }
        
        return {
            'forecast': forecast,
            'model': model,
            'metrics': metrics
        }

def plot_prophet_forecast(results, crypto_name):
    """
    Create an interactive plot for Prophet forecast
    
    Args:
        results (dict): Dictionary containing forecast and model
        crypto_name (str): Name of the cryptocurrency
    
    Returns:
        go.Figure: Plotly figure object
    """
    forecast = results['forecast']
    model = results['model']
    
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=(
            f'{crypto_name} Price Forecast',
            'Forecast Components'
        ),
        vertical_spacing=0.2
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Forecast',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line=dict(color='rgba(0,0,255,0.2)'),
            name='Upper Bound'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(0,0,255,0.2)'),
            name='Lower Bound'
        ),
        row=1, col=1
    )
    
    components = ['trend', 'weekly', 'yearly']
    colors = ['red', 'green', 'purple']
    
    for component, color in zip(components, colors):
        if component in forecast.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast[component],
                    name=f'{component.capitalize()} Component',
                    line=dict(color=color)
                ),
                row=2, col=1
            )
    
    fig.update_layout(
        height=800,
        template="plotly_dark",
        showlegend=True,
        title=dict(
            text=f"{crypto_name} Prophet Forecast Analysis",
            x=0.5
        )
    )
    
    return fig

def add_prophet_section(tabs, all_df, filenames, crypto_names):
    """Add Prophet model section to the app"""
    with tabs[1]:
        st.header("Prophet Model Predictions")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_crypto = st.selectbox(
                "Select Cryptocurrency for Prophet Analysis",
                crypto_names,
                key='prophet_crypto'
            )
            
            forecast_days = st.slider(
                "Forecast Days",
                min_value=7,
                max_value=90,
                value=30,
                key='prophet_forecast_days'
            )
        
        if st.button("Train Prophet Model"):
            try:
                # Get data for selected crypto
                symbol = filenames[crypto_names.index(selected_crypto)] + "/USD"
                crypto_df = all_df[all_df['symbol'] == symbol].copy()
                
                # Train Prophet model
                prophet_results = train_prophet_model(
                    crypto_df,
                    selected_crypto,
                    periods=forecast_days
                )
                
                # Display metrics
                with col2:
                    st.subheader("Prophet Model Metrics")
                    metrics = prophet_results['metrics']
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            st.metric(metric, f"{value:.2f}")
                        else:
                            st.metric(metric, value)
                
                # Plot forecast
                fig = plot_prophet_forecast(prophet_results, selected_crypto)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast table
                st.subheader("Detailed Forecast")
                forecast_df = prophet_results['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
                forecast_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                st.dataframe(forecast_df)
                
            except Exception as e:
                st.error(f"Error in Prophet analysis: {str(e)}")



class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


@st.cache_data
def load_data(urls, start_date = "2021-01-01"):
    filenames = []
    all_df = pd.DataFrame()
    
    for url in urls:
        req = requests.get(url, verify=False)
        filename = url[48:-9]
        filenames.append(filename)
        
        # Create DataFrame from content
        df = pd.read_csv(url, header=1, parse_dates=["date"])
        df = df[df["date"] > start_date]
        df.index = df.date
        df.drop(labels=[df.columns[0], df.columns[1], df.columns[8]], axis=1, inplace=True)
        all_df = pd.concat([all_df, df], ignore_index=False)
    
    return all_df, filenames

def computeRSI(data, time_window):
    diff = data.diff(1).dropna()
    up_chg = 0 * diff
    down_chg = 0 * diff
    up_chg[diff > 0] = diff[diff > 0]
    down_chg[diff < 0] = diff[diff < 0]
    up_chg_avg = up_chg.ewm(com=time_window-1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1, min_periods=time_window).mean()
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

def create_crypto_plot(df, crypto_name):
    fig = make_subplots(
        rows=3, 
        cols=2,
        shared_xaxes=True,
        specs=[[{"rowspan": 2}, {"rowspan": 2}], 
               [{"rowspan": 1}, {"rowspan": 1}],
               [{}, {}]]
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume USD"],
            showlegend=False,
            marker_color='aqua'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            showlegend=False,
            line=dict(color="red", width=4)
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['low'],
            fill='tonexty',
            mode='lines',
            showlegend=False,
            line=dict(width=2, color='pink', dash='dash')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['high'],
            fill='tonexty',
            mode='lines',
            showlegend=False,
            line=dict(width=2, color='pink', dash='dash')
        ),
        row=1, col=2
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close_rsi'],
            mode='lines',
            showlegend=False,
            line=dict(color="aquamarine", width=4)
        ),
        row=3, col=2
    )
    

    fig.update_layout(
        width=1120,
        height=650,
        template="plotly_dark",
        title=dict(
            text=f'<b>{crypto_name} Dashboard</b>',
            font=dict(color='#FFFFFF', size=22),
            x=0.5
        ),
        showlegend=False
    )
    

    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=True,
        gridcolor='#595959'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#595959',
        ticksuffix='$'
    )
    

    fig.update_yaxes(ticksuffix="", range=[0, 100], row=3, col=2)
    
    return fig


def create_technical_indicators(df):
    """Ajoute les indicateurs techniques"""
    try:
        # RSI
        df['close_rsi'] = computeRSI(df['close'], time_window=RSI_TIME_WINDOW)
        df['high_rsi'] = 70
        df['low_rsi'] = 30
        
        # Moyennes mobiles
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        
        # Volatilité
        df['volatility'] = df['high'] - df['low']
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du calcul des indicateurs: {str(e)}")
        return df

def prepare_data(df, sequence_length=30, prediction_days=7, train_split=0.8):
    """Prépare les données pour l'entraînement"""
    if df.empty:
        raise ValueError("DataFrame vide")
    

    returns = df['close'].pct_change()
    

    mean_return = returns.mean()
    std_return = returns.std()
    returns = returns[abs(returns - mean_return) <= 3 * std_return]
    

    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns) < sequence_length + prediction_days + 1:
        raise ValueError(f"Pas assez de données: {len(returns)} points disponibles")
    

    scaler = MinMaxScaler()
    returns_scaled = scaler.fit_transform(returns.values.reshape(-1, 1))
    

    X, y = [], []
    for i in range(len(returns_scaled) - sequence_length - prediction_days):
        X.append(returns_scaled[i:(i + sequence_length)])
        y.append(returns_scaled[i + sequence_length:i + sequence_length + prediction_days])
    
    X = np.array(X)
    y = np.array(y)
    

    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(model, train_loader, test_loader, epochs, device):
    """Entraîne le modèle LSTM"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses = []
    test_losses = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart = st.empty()
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.squeeze(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.squeeze(-1))
                test_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        
        scheduler.step(avg_test_loss)
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            st.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
        

        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
        
        loss_df = pd.DataFrame({
            'Training Loss': train_losses,
            'Validation Loss': test_losses
        })
        loss_chart.line_chart(loss_df)
    
    return model, train_losses, test_losses

def evaluate_predictions(model, X_test, y_test, scaler, device):
    """Évalue les prédictions du modèle"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_tensor).cpu().numpy()
        
        predictions = scaler.inverse_transform(predictions)
        y_test_transformed = scaler.inverse_transform(y_test.squeeze(-1))
        
        mse = np.mean((predictions - y_test_transformed) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test_transformed))
        
        r2 = 1 - np.sum((y_test_transformed - predictions) ** 2) / np.sum((y_test_transformed - np.mean(y_test_transformed)) ** 2)
        
        correct_direction = np.sum(np.sign(predictions[1:] - predictions[:-1]) == 
                                 np.sign(y_test_transformed[1:] - y_test_transformed[:-1]))
        direction_accuracy = correct_direction / (len(predictions) - 1)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Direction Accuracy': direction_accuracy
        }
        
        return predictions, y_test_transformed, metrics
    
def train_all_models(all_df, filenames, crypto_names, params, device):
    """Entraîne un modèle pour chaque crypto"""
    all_predictions = {}
    all_metrics = {}
    all_returns = pd.DataFrame()
    
    col1, col2 = st.columns(2)
    
    for idx, (filename, crypto_name) in enumerate(zip(filenames, crypto_names)):
        with col1:
            st.write(f"Training model for {crypto_name}...")
        
        try:
            symbol = filename + "/USD"
            crypto_df = all_df[all_df['symbol'] == symbol].copy()
            
            if crypto_df.empty:
                st.warning(f"No data available for {crypto_name}")
                continue
            
            returns = crypto_df['close'].pct_change().dropna()
            all_returns[crypto_name] = returns
            
            X_train, X_test, y_train, y_test, scaler = prepare_data(
                crypto_df,
                sequence_length=params['sequence_length'],
                prediction_days=params['prediction_days']
            )
            
            train_dataset = CryptoDataset(X_train, y_train)
            test_dataset = CryptoDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])
            
            model = LSTM(
                input_size=1,
                hidden_size=64,
                num_layers=2,
                output_size=params['prediction_days'],
                dropout=0.2
            ).to(device)
            
            model, _, _ = train_model(model, train_loader, test_loader, params['epochs'], device)
            predictions, actual, metrics = evaluate_predictions(model, X_test, y_test, scaler, device)
            
            all_predictions[crypto_name] = predictions
            all_metrics[crypto_name] = metrics
            
            with col2:
                st.write(f"Metrics for {crypto_name}:")
                st.write(f"RMSE: {metrics['RMSE']:.6f}")
                st.write(f"Direction Accuracy: {metrics['Direction Accuracy']*100:.2f}%")
                st.write("---")
                
        except Exception as e:
            st.error(f"Error training model for {crypto_name}: {str(e)}")
            continue
    
    return all_predictions, all_metrics, all_returns

def optimize_portfolio(returns_df, risk_free_rate=0.01):
    """Optimise le portefeuille en utilisant le ratio de Sharpe"""
    def negative_sharpe(weights):
        portfolio_returns = returns_df.dot(weights)
        expected_return = portfolio_returns.mean() * 252
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = (expected_return - risk_free_rate) / portfolio_vol
        return -sharpe
    
    n_assets = len(returns_df.columns)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = optimize.minimize(
        negative_sharpe,
        x0=np.array([1/n_assets] * n_assets),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

def plot_portfolio_allocation(weights, crypto_names):
    """Visualise l'allocation du portefeuille"""
    fig = go.Figure(data=[go.Pie(
        labels=crypto_names,
        values=weights,
        hole=.3,
        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    )])
    
    fig.update_layout(
        title="Optimal Portfolio Allocation",
        template="plotly_dark"
    )
    
    return fig

def plot_crypto_dashboard(df, crypto_name):
    """Crée un dashboard pour une crypto"""
    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=True,
        specs=[[{"rowspan": 2}, {"rowspan": 2}],
               [{"rowspan": 1}, {"rowspan": 1}],
               [{}, {}]],
        subplot_titles=(
            'Candlestick Chart', 'Price Chart',
            '', '',
            'Volume USD', 'RSI'
        )
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        ),
        row=1, col=1
    )
    
    # Price
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close'],
            name='Price',
            line=dict(color="red")
        ),
        row=1, col=2
    )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume USD'],
            name='Volume',
            marker_color='aqua'
        ),
        row=3, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close_rsi'],
            name='RSI',
            line=dict(color="purple")
        ),
        row=3, col=2
    )
    
    fig.update_layout(
        height=800,
        template="plotly_dark",
        title=f"{crypto_name} Analysis"
    )
    
    return fig

def calculate_portfolio_metrics(weights, returns_df):
    """Calcule les métriques du portefeuille"""
    portfolio_returns = returns_df.dot(weights)
    
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol
    max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()
    
    return {
        "Annual Return": f"{annual_return*100:.2f}%",
        "Annual Volatility": f"{annual_vol*100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Maximum Drawdown": f"{max_drawdown*100:.2f}%"
    }

def main():
    st.title("Cryptocurrency &Stocks  Analysis and Portfolio Optimization")
    tabs = st.tabs(["Cryptocurrency", "Stocks"])
    with tabs[1]:
        st.subheader("Stocks Analysis")
        st.title("Stock Forecasting & Analysis")
        sk_tabs = st.tabs(["Time Series Analysis", "Forecasting"])
        with sk_tabs[0]:
            st.subheader("Stock Time Series Analysis")
            company = st.selectbox("Select Company", ["AAPL - Apple", "NVDA - Nvidia", "FB - Facebook (Meta)", "TSLA - Tesla"])
            tickers = {
                "AAPL - Apple": "AAPL",
                "NVDA - Nvidia": "NVDA",
                "FB - Facebook (Meta)": "META",
                "TSLA - Tesla": "TSLA"
            }
            start_date = st.date_input("Select Start Date", value=datetime(2020, 1, 1))
            end_date = st.date_input("Select End Date", value=datetime.now())
            if 'stock_data' not in st.session_state:
                st.session_state['stock_data'] = None

            if st.button("Show Time Series Data"):
                st.session_state['stock_data'] = load_yfinance_data(tickers[company], start_date, end_date)
                st.session_state['stock_data_adj_close'] = st.session_state['stock_data']['Adj Close'].astype(np.float32)

                st.subheader(f"Historical Data - {company}")
                st.session_state['time_series_fig'] = plot_time_series(st.session_state['stock_data_adj_close'], f"Adjusted Stock Prices ({start_date} - {end_date})", "Adjusted Price (USD)", key="time_series")
                st.session_state['log_returns_fig'], log_returns = plot_log_returns(st.session_state['stock_data_adj_close'], key="log_returns")
                st.subheader("Stationarity Test (ADF)")
                adf_test(st.session_state['stock_data_adj_close'])
                st.subheader("ACF and PACF")
                st.session_state['acf_pacf_fig'] = plot_acf_pacf(log_returns)
            if st.session_state['stock_data'] is not None:
                stock_data_adj_close = st.session_state['stock_data_adj_close']
                if 'time_series_fig' in st.session_state:
                    st.plotly_chart(st.session_state['time_series_fig'], key="time_series_reuse")
                if 'log_returns_fig' in st.session_state:
                    st.plotly_chart(st.session_state['log_returns_fig'], key="log_returns_reuse")

        with sk_tabs[1]:
            model_choice = st.radio("Select Forecasting Model", ["SARIMA", "Prophet"])
            forecast_steps = st.number_input("Forecast Steps", min_value=1, max_value=365, value=30)
            if model_choice == "SARIMA":
                param_choice = st.radio("Choisir la méthode", ("Auto-ARIMA (optimisation automatique)", "Paramètres manuels"))

                if param_choice == "Paramètres manuels":
                    p = st.number_input("Paramètre p (ordre AR)", min_value=0, max_value=5, value=1)
                    d = st.number_input("Paramètre d (ordre de différenciation)", min_value=0, max_value=2, value=1)
                    q = st.number_input("Paramètre q (ordre MA)", min_value=0, max_value=5, value=1)

                    seasonal = st.checkbox("Inclure des composantes saisonnières (SARIMA)")
                    if seasonal:
                        P = st.number_input("Paramètre P (ordre AR saisonnier)", min_value=0, max_value=5, value=1)
                        D = st.number_input("Paramètre D (ordre de différenciation saisonnier)", min_value=0, max_value=2, value=1)
                        Q = st.number_input("Paramètre Q (ordre MA saisonnier)", min_value=0, max_value=5, value=1)
                        s = st.number_input("Paramètre s (période saisonnière)", min_value=1, max_value=365, value=12)

                if param_choice == "Auto-ARIMA (optimisation automatique)":
                    if st.button("Lancer la prévision avec SARIMA"):
                        st.subheader("Optimisation des paramètres avec auto-ARIMA")
                        stock_data_log = np.log(stock_data_adj_close)  
                        auto_model = auto_arima(stock_data_log, start_p=1, start_q=1,
                                                max_p=5, max_q=5, m=12,
                                                start_P=0, seasonal=True,
                                                d=1, D=1, trace=True,
                                                error_action='ignore',
                                                suppress_warnings=True,
                                                stepwise=True)

                        st.write(auto_model.summary())
                        model = SARIMAX(stock_data_log, order=auto_model.order, seasonal_order=auto_model.seasonal_order)
                        results = model.fit()

                        forecast = results.get_forecast(steps=forecast_steps)
                        forecast_values = np.exp(forecast.predicted_mean)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=stock_data_adj_close.index, y=stock_data_adj_close, mode='lines', name='Données Historiques'))
                        fig.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values, mode='lines', name='Prévisions'))
                        fig.update_layout(title=f'Prévisions SARIMA (Auto-ARIMA) pour {forecast_steps} jours', xaxis_title='Date', yaxis_title='Prix Ajusté (USD)')
                        st.plotly_chart(fig, key='sarima_forecast')

                elif param_choice == "Paramètres manuels":
                    if st.button("Lancer la prévision avec les paramètres manuels"):
                        stock_data_log = np.log(stock_data_adj_close)  
                        if seasonal:
                            model = SARIMAX(stock_data_log, order=(p, d, q), seasonal_order=(P, D, Q, s))
                        else:
                            model = SARIMAX(stock_data_log, order=(p, d, q))

                        results = model.fit()

                        forecast = results.get_forecast(steps=forecast_steps)
                        forecast_values = np.exp(forecast.predicted_mean)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=stock_data_adj_close.index, y=stock_data_adj_close, mode='lines', name='Données Historiques'))
                        fig.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values, mode='lines', name='Prévisions'))
                        fig.update_layout(title=f'Prévisions SARIMA (paramètres manuels) pour {forecast_steps} jours', xaxis_title='Date', yaxis_title='Prix Ajusté (USD)')
                        st.plotly_chart(fig, key='sarima_manual_forecast')

            elif model_choice == "Prophet":
                    if st.button("Lancer la prévision avec Prophet"):
                        run_prophet(stock_data_adj_close, forecast_steps)

    with tabs[0]:
        st.subheader("Cryptocurrency Analysis")
        tabs = st.tabs(["Data Analysis", "LSTM Training","Facebook Prophet training", "Portfolio Optimization"])
        
        with st.spinner("Loading data..."):
            try:
                all_df, filenames = load_data(urls, START_DATE)
                st.success("Data loaded successfully!")
                returns_df = pd.DataFrame()
                
                for filename, crypto_name in zip(filenames, crypto_names):
                    symbol = filename + "/USD"
                    crypto_df = all_df[all_df['symbol'] == symbol].copy()
                    returns_df[crypto_name] = crypto_df['close'].pct_change().dropna()
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return

        with tabs[0]:
            st.subheader("Technical Analysis")
            selected_crypto = st.selectbox("Select Cryptocurrency", crypto_names)
            
            symbol = filenames[crypto_names.index(selected_crypto)] + "/USD"
            crypto_df = all_df[all_df['symbol'] == symbol].copy()
            
            crypto_df = create_technical_indicators(crypto_df)
            fig = plot_crypto_dashboard(crypto_df, selected_crypto)
            st.plotly_chart(fig)
        with tabs[1]:
            crypto_model_choice = st.radio("Select a Model for Cryptocurrency Forecasting", ("Technical Analysis", "LSTM"))
            if crypto_model_choice == "LSTM":
                st.sidebar.header("LSTM Parameters")
                params = {
                    'sequence_length': st.sidebar.slider("Sequence Length", 10, 50, 30),
                    'prediction_days': st.sidebar.slider("Prediction Days", 1, 30, 7),
                    'epochs': st.sidebar.slider("Training Epochs", 10, 100, 50),
                    'batch_size': st.sidebar.slider("Batch Size", 16, 64, 32)
                }

                risk_free_rate = st.sidebar.number_input("Risk-free Rate (%)", 0.0, 10.0, 1.0) / 100

            if st.button("Train Models"):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                st.info(f"Using device: {device}")
                
                try:
                    all_predictions, all_metrics, model_returns = train_all_models(
                        all_df, filenames, crypto_names, params, device
                    )
                    
                    st.session_state['all_predictions'] = all_predictions
                    st.session_state['all_metrics'] = all_metrics
                    st.session_state['all_returns'] = model_returns
                    
                    st.success("Training completed!")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        
        with tabs[3]:
            if 'all_returns' not in st.session_state:
                st.warning("Please train the models first.")
            else:
                try:
                    optimal_weights = optimize_portfolio(st.session_state['all_returns'], risk_free_rate)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = plot_portfolio_allocation(optimal_weights, crypto_names)
                        st.plotly_chart(fig)
                    
                    with col2:
                        metrics = calculate_portfolio_metrics(
                            optimal_weights, 
                            st.session_state['all_returns']
                        )
                        st.write("Portfolio Metrics:")
                        for metric, value in metrics.items():
                            st.metric(metric, value)
                    
                except Exception as e:
                    st.error(f"Error in portfolio optimization: {str(e)}")
        with tabs[2]:
            st.header("Prophet Model Predictions")
            col1, col2 = st.columns(2)
            
            with col1:
                selected_crypto = st.selectbox(
                    "Select Cryptocurrency for Prophet Analysis",
                    crypto_names,
                    key='prophet_crypto'
                )
                
                forecast_days = st.slider(
                    "Forecast Days",
                    min_value=7,
                    max_value=90,
                    value=30,
                    key='prophet_forecast_days'
                )
            
            if st.button("Generate Prophet Forecast"):
                try:
                    symbol = filenames[crypto_names.index(selected_crypto)] + "/USD"
                    crypto_df = all_df[all_df['symbol'] == symbol].copy()
                    
    
                    prophet_results = train_prophet_model(
                        crypto_df,
                        selected_crypto,
                        periods=forecast_days
                    )
                    
                    with col2:
                        st.subheader("Prophet Model Metrics")
                        metrics = prophet_results['metrics']
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                st.metric(metric, f"{value:.2f}")
                            else:
                                st.metric(metric, value)
                    
                    fig = plot_prophet_forecast(prophet_results, selected_crypto)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Detailed Forecast")
                    forecast_df = prophet_results['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
                    forecast_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                    st.dataframe(forecast_df)
                    if st.checkbox("Show Forecast Components"):
                        st.subheader("Forecast Components")
                        try:
                            fig_comp = prophet_results['model'].plot_components(prophet_results['forecast'])
                            st.pyplot(fig_comp)
                        except Exception as e:
                            st.error(f"Error displaying components: {str(e)}")
                    
                except Exception as e:
                    st.error(f"Error in Prophet analysis: {str(e)}")

if __name__ == "__main__":
    main()
