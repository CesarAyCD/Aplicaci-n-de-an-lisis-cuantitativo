import pandas as pd
import numpy as np

TARGET = 10
TARGET_METHOD = "triple_barrier"
TRIPLE_BARRIER_PT = 1.5
TRIPLE_BARRIER_SL = 1.0
TRIPLE_BARRIER_VOL_SPAN = 50
TRIPLE_BARRIER_MIN_VOL_PERIODS = 20

# Añade las etiquetas necesarias para el método de backtest que comparte nombre
def triple_barrier_labeling(close, high, low, horizon=TARGET, pt_mult=TRIPLE_BARRIER_PT,
                            sl_mult=TRIPLE_BARRIER_SL,
                            vol_span=TRIPLE_BARRIER_VOL_SPAN,
                            min_vol_periods=TRIPLE_BARRIER_MIN_VOL_PERIODS):
    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    volatility = np.log(close / close.shift(1)).ewm(
        span=vol_span, min_periods=min_vol_periods
    ).std()

    prices = close.to_numpy()
    highs = high.to_numpy()
    lows = low.to_numpy()
    vols, n = volatility.to_numpy(), len(close)

    labels = np.full(n, np.nan)
    returns = np.full(n, np.nan)
    touches = np.full(n, np.nan)

    # Simula la trayectoria de precios
    for i in range(n - horizon):
        if not np.isfinite(vols[i]) or prices[i] <= 0:
            continue
        close_path = np.log(prices[i + 1:i + horizon + 1] / prices[i])
        high_path = np.log(highs[i + 1:i + horizon + 1] / prices[i])
        low_path = np.log(lows[i + 1:i + horizon + 1] / prices[i])

        # Encuentra dónde tocaría el take profit o stop loss
        upper = np.flatnonzero(high_path >= pt_mult * vols[i])
        lower = np.flatnonzero(low_path <= -sl_mult * vols[i])

        up_bar = upper[0] + 1 if upper.size else np.inf
        down_bar = lower[0] + 1 if lower.size else np.inf

        # Asigna las señales dinámicas para take profit y stop loss 
        if up_bar < down_bar:
            touch, label, target_return = int(up_bar), 1, pt_mult * vols[i]
        elif np.isfinite(down_bar) and down_bar <= up_bar:
            touch, label, target_return = int(down_bar), -1, -sl_mult * vols[i]
        # Si no consigue ninguna señal, asigna un límite de tiempo a la posición
        else:
            touch = horizon
            label = int(np.sign(close_path[-1]))
            target_return = close_path[-1]
        labels[i], returns[i], touches[i] = label, target_return, touch

    return pd.DataFrame({
        "triple_barrier_label": labels,
        "target_return": returns,
        "barrier_touch_bars": touches,
        "target_volatility": volatility,
    }, index=close.index)


def calcula_atr(df, length):
    data = df.copy()
    h_l = data["high"] - data["low"]
    h_c_prev = np.abs(data["high"] - data["close"].shift(1))
    l_c_prev = np.abs(data["low"] - data["close"].shift(1))
    
    data["tr"] = np.maximum(h_l, np.maximum(h_c_prev, l_c_prev))   
    data["atr_14"] = data["tr"].ewm(alpha=1 / length, adjust = False).mean()
    data["atr_14_pct"] = data["atr_14"] / data["close"]
    
    return data["atr_14_pct"]

def calcula_rsi(df, length):
    delta = df["close"].diff()
    gain = delta.clip(lower = 0)
    loss = -delta.clip(upper = 0)

    avg_gain = gain.ewm(alpha = 1 / length, adjust = False).mean()
    avg_loss = loss.ewm(alpha = 1 / length, adjust = False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calcula_bollinger(df, length):
    bol_sma = df["close"].rolling(window = length).mean()
    bol_std = df["close"].rolling(window = length).std()

    upper = bol_sma + 2 * bol_std
    lower = bol_sma - 2 * bol_std

    return (df["close"] - lower) / (upper - lower), (upper - lower) / bol_sma


def session_features(df):
    features = pd.DataFrame(index=df.index)
    features["bars_in_session"] = df["bar_in_session"]
    features["n_minutes"] = df["n_minutes"]
    features["day_of_week"] = df["day_of_week"]
    features["opening_bar"] = (df["bar_in_session"] == 0).astype(int)
    features["closing_bar"] = (
        df.groupby("session_date").cumcount(ascending=False) == 0
    ).astype(int)
    return features

# Hace un preprocesamiento completo a los datos obtenidos de IBKR para que sean compatibles con el diseño del sistema

def price_features(df):
    features = pd.DataFrame(index=df.index)
    features["log_return"] = np.log(df["close"] / df["close"].shift(1))
    features["open_close_return"] = (df["close"] - df["open"]) / df["open"]
    features["high_low_range_pct"] = (df["high"] - df["low"]) / df["low"]
    features["close_position"] = ((df["close"] - df["low"]) / (df["high"] - df["low"]))

    for window in (3, 6, 12, 21):
        features[f"return_{window}"] = np.log(
            df["close"] / df["close"].shift(window)
        )

    for window in (6, 21, 42):
        features[f"volatility_{window}"] = features["log_return"].rolling(window).std()
    features["atr_14_pct"] = calcula_atr(df, 14)

    return features


def volume_features(df):
    features = pd.DataFrame(index=df.index)
    rolling_volume = df["volume"].rolling(21)
    features["log_volume"] = np.log1p(df["volume"])
    features["relative_volume_21"] = df["volume"] / rolling_volume.mean()
    features["volume_zscore_21"] = ((df["volume"] - rolling_volume.mean()) / rolling_volume.std())
    # Normaliza el volumen por franja horaria, ya que se suele observar un volumen mayor al final de las
    # sesiones, intentamos no sesgar el modelo
    historical_slot_volume = df.groupby("bar_in_session")["volume"].transform(
        lambda values: values.shift(1).rolling(50, min_periods=10).mean()
    )
    features["relative_volume_by_hour"] = df["volume"] / historical_slot_volume

    return features


def liquidity_features(df):
    features = pd.DataFrame(index=df.index)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    rolling_count = df["barCount"].rolling(21)
    positive_count = df["barCount"].replace(0, np.nan)

    features["average_close_deviation"] = df["average"] / df["close"] - 1
    features["average_position_in_range"] = (
        (df["average"] - df["low"]) / candle_range
    )

    # Se usa barcount para estimar la intensidad de negociación
    features["log_bar_count"] = np.log1p(df["barCount"])
    features["relative_bar_count_21"] = df["barCount"] / rolling_count.mean()
    features["bar_count_zscore_21"] = (
        (df["barCount"] - rolling_count.mean()) / rolling_count.std()
    )
    features["average_volume_per_trade"] = df["volume"] / positive_count
    return features


def technical_features(df):
    features = pd.DataFrame(index=df.index)
    for window in (21, 42, 120):
        features[f"dist_sma_{window}"] = (df["close"] / df["close"].rolling(window).mean() - 1)
    features["rsi_14"] = calcula_rsi(df, 14)
    features["bollinger_position"], features["bollinger_width"] = (calcula_bollinger(df, 20))
    return features

# Crea las variables objetivo para los modelos de ML
def target_features(df):
    if TARGET_METHOD == "triple_barrier":
        return triple_barrier_labeling(df["close"], df["high"], df["low"])
    if TARGET_METHOD == "fixed_horizon":
        target_return = np.log(df["close"].shift(-TARGET) / df["close"])
        return pd.DataFrame({
            "target_return": target_return,
            "triple_barrier_label": np.sign(target_return),
            "barrier_touch_bars": TARGET,
        }, index=df.index)
    raise ValueError(f"TARGET_METHOD no reconocido: {TARGET_METHOD}")

# Función principal. Sirve como orquestador
def preprocesamiento(df):
    frames = [
        session_features(df), price_features(df), volume_features(df),
        liquidity_features(df), technical_features(df), target_features(df),
    ]
    features = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    # Se conservan para simular las barreras en el backtest, pero train_models las excluye
    features["close"] = df.loc[features.index, "close"]
    features["high"] = df.loc[features.index, "high"]
    features["low"] = df.loc[features.index, "low"]
    features["target_direction"] = (
        features["triple_barrier_label"] > 0
    ).astype(int)
    return features
