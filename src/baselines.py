import pandas as pd

# Usa estrategias tradicionales para crear señales de compra
def make_baseline_signals(df):
    signals = {}

    # No es una señal predictiva, solo sirve como benchmark
    signals["always_long"] = pd.Series(1, index=df.index)

    # Momentum
    signals["momentum_strategy"] = (
        (df["return_3"] > 0)
        & (df["return_6"] > 0)
        & (df["dist_sma_42"] > 0)
    ).astype(int)

    # Trend following:
    signals["trend_following_strategy"] = (
        (df["dist_sma_42"] > 0)
        & (df["dist_sma_120"] > 0)
        & (df["rsi_14"].between(50, 70))
    ).astype(int)

    # Pullback en tendencia:
    signals["trend_pullback_strategy"] = (
        (df["dist_sma_120"] > 0)
        & (df["return_3"] < 0)
        & (df["bollinger_position"] < 0.45)
        & (df["rsi_14"].between(35, 55))
    ).astype(int)

    # Reversión a la media:
    signals["mean_reversion_strategy"] = (
        (df["return_3"] < 0)
        & (df["bollinger_position"] < 0.25)
        & (df["rsi_14"] < 35)
        & (df["log_return"] > 0)
    ).astype(int)

    # RSI oversold:
    signals["rsi_oversold_strategy"] = (
        (df["rsi_14"] < 30)
        & (df["log_return"] > 0)
    ).astype(int)

    # RSI momentum:
    signals["rsi_momentum_strategy"] = (
        (df["rsi_14"] >= 50)
        & (df["rsi_14"] < 70)
        & (df["return_3"] > 0)
    ).astype(int)

    # Bollinger mean reversion:
    signals["bollinger_reversion_strategy"] = (
        (df["bollinger_position"] < 0.20)
        & (df["log_return"] > 0)
    ).astype(int)

    # Bollinger breakout:
    signals["bollinger_breakout_strategy"] = (
        (df["bollinger_position"] > 0.80)
        & (df["return_3"] > 0)
        & (df["rsi_14"] >= 50)
    ).astype(int)

    # Breakout con volumen:
    historical_range = df["high_low_range_pct"].shift(1).rolling(21).median()

    signals["volume_breakout_strategy"] = (
        (df["return_3"] > 0)
        & (df["high_low_range_pct"] > historical_range)
        & (df["relative_volume_21"] > 1.10)
        & (df["close_position"] > 0.60)
    ).astype(int)

    return signals

# Carga las señales a los modelos
def add_baseline_signal_features(df):
    output = df.copy()

    for name, signal in make_baseline_signals(output).items():
        if name == "always_long":
            continue
        output[f"baseline_signal_{name}"] = signal.astype(int)

    return output