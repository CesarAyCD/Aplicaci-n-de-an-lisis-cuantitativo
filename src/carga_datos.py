from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
# Columnas necesarias para el correcto funcionamiento del sistema.
# Son las columnas que por defecto entrega IBKR. 
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume", "average", "barCount"}

# Convierte la temporalidad al número de minutos, que después se usa para dar contexto
# sobre el verdadero significado del tamaño de las velas al modelo
def timeframe_to_minutes(timeframe):
    units = {
        "s": 1 / 60,
        "m": 1,
        "h": 60,
        "d": 24 * 60,
        "w": 7 * 24 * 60,
        "mo": 30 * 24 * 60,
    }

    for suffix in ("mo", "s", "m", "h", "d", "w"):
        if timeframe.endswith(suffix):
            value = timeframe[:-len(suffix)]
            if value.isdigit() and int(value) > 0:
                return int(value) * units[suffix]
    raise ValueError(f"Temporalidad no reconocida: {timeframe}")

# Limpia, indexa y añade features al df. En principio, no es necesaria la validación de duplicados (por ejemplo)
# Esto se aplica por el caso en que se use el código proporcionado para descargar y almacenar datos
def preparar_datos_ibkr(df, ticker, timeframe):
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{ticker}: faltan columnas: {', '.join(sorted(missing))}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data["date"].dt.tz is not None:
        data["date"] = data["date"].dt.tz_localize(None)

    data = (
        data.drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .set_index("date")
    )
    data.index.name = "date"
    data["session_date"] = data.index.date
    data["bar_in_session"] = data.groupby("session_date").cumcount()
    data["n_minutes"] = timeframe_to_minutes(timeframe)
    data["hour"] = data.index.hour
    data["minute"] = data.index.minute
    data["day_of_week"] = data.index.dayofweek
    data["ticker"] = ticker
    data["timeframe"] = timeframe
    return data

# Carga y procesa los datos guardados (.parquet)
# Si no se usan los filtros lee todos los archivos de ./data 
def bases_datos(tickers=None, timeframes=None, data_dir=DEFAULT_DATA_DIR):
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*_*_history.parquet"))
    ticker_filter = set(tickers) if tickers else None
    timeframe_filter = set(timeframes) if timeframes else None

    if not paths:
        raise FileNotFoundError(
            f"No hay archivos *_history.parquet en {data_dir}. "
            "Ejecuta primero ./IB_main.py."
        )

    datasets = {}
    for path in paths:
        ticker, timeframe = path.name.removesuffix("_history.parquet").rsplit("_", 1)
        if ticker_filter and ticker not in ticker_filter:
            continue
        if timeframe_filter and timeframe not in timeframe_filter:
            continue
        data = preparar_datos_ibkr(pd.read_parquet(path), ticker, timeframe)
        datasets[(ticker, timeframe)] = data
        print(f"Cargado {ticker} {timeframe}: {len(data)} filas")
    if not datasets:
        raise FileNotFoundError("No hay históricos que coincidan con los filtros.")
    return datasets
