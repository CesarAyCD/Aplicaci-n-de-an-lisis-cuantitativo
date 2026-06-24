import numpy as np
import pandas as pd
from feature_eng import calcula_rsi

# Asigna a cada ticker su ETF sectorial
SECTOR_ETF_BY_TICKER = {
    "AAPL": "XLK", "AMD": "XLK", "MSFT": "XLK", "NVDA": "XLK", "ORCL": "XLK", "CRM": "XLK",
    "IBM": "XLK", "INTC": "XLK", "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "COST": "XLP", "WMT": "XLP",
    "KO": "XLP", "PEP": "XLP", "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "XOM": "XLE",
    "CVX": "XLE", "GE": "XLI", "BA": "XLI", "CAT": "XLI", "LLY": "XLV", "UNH": "XLV",
    "JNJ": "XLV", "PFE": "XLV",
}

SECTOR_ETFS = {
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLRE", "XLU", "XLV", "XLY",
}

def log_return(close, periods=1):
    return np.log(close / close.shift(periods))

def availability_index(index, minutes):
    return index + pd.to_timedelta(float(minutes), unit="m")

def median_minutes(df):
    return float(df["n_minutes"].median())

# Combina datos de forma asíncrona hacia atrás. Garantiza que no haya data leaking
def merge_asof_available(features, context, source_minutes):
    if context.empty:
        return pd.DataFrame(index=features.index)

    current_minutes = median_minutes(features)
    left = pd.DataFrame({
        "_row": np.arange(len(features)),
        "_available_at": availability_index(features.index, current_minutes),
    })
    right = context.copy()
    right["_available_at"] = availability_index(context.index, source_minutes)
    right = right.sort_values("_available_at").reset_index(drop=True)

    merged = pd.merge_asof(
        left.sort_values("_available_at"),
        right,
        on="_available_at",
        direction="backward",
    ).sort_values("_row")

    merged.index = features.index
    return merged.drop(columns=["_row", "_available_at"])

# Usa datos intradía para crear velas diarias
def daily_ohlcv(raw):
    if "session_date" in raw:
        daily = raw.groupby("session_date").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        daily.index = pd.to_datetime(daily.index)
        return daily.sort_index()
    return raw[["open", "high", "low", "close", "volume"]].copy().sort_index()

# Crea contexto usando velas diarias
def daily_context(raw):
    daily = daily_ohlcv(raw)
    daily_return = log_return(daily["close"])
    sma_20 = daily["close"].rolling(20).mean()
    sma_50 = daily["close"].rolling(50).mean()

    context = pd.DataFrame(index=daily.index)
    context["return_1d"] = daily_return
    context["volatility_1d"] = daily_return.rolling(21).std()
    context["dist_sma_daily_20"] = daily["close"] / sma_20 - 1
    context["rsi_daily_14"] = calcula_rsi(daily, 14)
    context["trend_daily"] = np.sign(sma_20 / sma_50 - 1)
    return context.shift(1)

# Envía el contexto diario a temporalidades más bajas
def add_daily_context(features, raw):
    context = daily_context(raw)
    session_dates = pd.to_datetime(pd.Series(features.index.date, index=features.index))
    for column in context.columns:
        features[column] = session_dates.map(context[column])
    return features

# Es la misma lógica anterior, pero esta vez por hora
def hourly_context(raw):
    context = pd.DataFrame(index=raw.index)
    context["return_1h"] = log_return(raw["close"])
    return context

def add_hourly_context(features, ticker, timeframe, raw_datasets):
    hourly = raw_datasets.get((ticker, "1h"))
    if hourly is None:
        return features
    if median_minutes(features) >= median_minutes(hourly):
        return features
    context = merge_asof_available(features, hourly_context(hourly), median_minutes(hourly))
    for column in context.columns:
        features[column] = context[column]
    return features

# Mapea el activo a su ETF
def sector_etf_for(ticker):
    if ticker in SECTOR_ETFS:
        return ticker
    if ticker in {"SPY", "QQQ", "DIA", "IWM", "MDY", "VTI"}:
        return "SPY"
    if ticker in {"EEM", "EFA"}:
        return "SPY"
    if ticker in {"TLT", "IEF", "SHY", "LQD", "HYG"}:
        return "TLT"
    if ticker in {"GLD", "SLV", "USO"}:
        return ticker
    if ticker in {"VNQ", "XLRE"}:
        return "XLRE"
    if ticker == "ARKK":
        return "QQQ"
    return SECTOR_ETF_BY_TICKER.get(ticker, "SPY")

# Calcula momentum y volatilidad para el mercado
def market_return_context(raw, prefix):
    context = pd.DataFrame(index=raw.index)
    for window in (3, 21):
        context[f"{prefix}_return_{window}"] = log_return(raw["close"], window)
    context[f"{prefix}_volatility_60"] = log_return(raw["close"]).rolling(60).std()
    return context

# Sincroniza índices con activos
def add_market_series(features, raw, source, prefix):
    context = merge_asof_available(
        features,
        market_return_context(source, prefix),
        median_minutes(source),
    )
    for column in context.columns:
        features[column] = context[column]
    return features

# Añade el comportamiento reciente del sector
def add_sector_context(features, source):
    context = pd.DataFrame(index=source.index)
    context["sector_etf_return_21"] = log_return(source["close"], 21)
    context = merge_asof_available(features, context, median_minutes(source))
    features["sector_etf_return_21"] = context["sector_etf_return_21"]
    return features

# Añade métricas comparativas del sector contra S&P
def add_relative_market_features(features, raw, spy):
    ticker_return = log_return(raw["close"])
    spy_return = log_return(spy["close"]).reindex(raw.index)
    spy_return = spy_return.ffill()
    ticker_volatility = ticker_return.rolling(60).std()
    spy_volatility = spy_return.rolling(60).std()

    relative = pd.DataFrame(index=raw.index)
    for window in (3, 21):
        relative[f"ticker_minus_SPY_return_{window}"] = (
            log_return(raw["close"], window)
            - log_return(spy["close"], window).reindex(raw.index).ffill()
        )
    relative["ticker_volatility_spy_ratio_60"] = ticker_volatility / spy_volatility
    relative["beta_rolling_60"] = (
        ticker_return.rolling(60).cov(spy_return) / spy_return.rolling(60).var()
    )
    relative["correlation_SPY_60"] = ticker_return.rolling(60).corr(spy_return)

    aligned = merge_asof_available(features, relative, median_minutes(raw))
    for column in aligned.columns:
        features[column] = aligned[column]
    return features

# Función principal. Sirve como orquestador.
def add_context_features(features, ticker, timeframe, raw, raw_datasets):
    features = features.copy()
    features = add_hourly_context(features, ticker, timeframe, raw_datasets)

    daily_raw = raw_datasets.get((ticker, "1d"), raw)
    features = add_daily_context(features, daily_raw)

    spy = raw_datasets.get(("SPY", timeframe))
    if spy is not None:
        features = add_market_series(features, raw, spy, "SPY")
        features = add_relative_market_features(features, raw, spy)

    qqq = raw_datasets.get(("QQQ", timeframe))
    if qqq is not None:
        features = add_market_series(features, raw, qqq, "QQQ")

    sector_ticker = sector_etf_for(ticker)
    sector = raw_datasets.get((sector_ticker, timeframe))
    if sector is not None:
        features = add_sector_context(features, sector)

    return features
