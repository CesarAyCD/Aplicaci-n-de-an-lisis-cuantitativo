from datetime import timedelta
from pathlib import Path

import pandas as pd
from ib_async import util

BAR_COLUMNS = [
    "date", "open", "high", "low", "close", "volume", "average", "barCount"
]
DATA_DIR = Path(__file__).resolve().parent.parent


def _timeframe_slug(bar_size):
    value, unit = bar_size.lower().split(maxsplit=1)
    suffixes = {
        "sec": "s", "secs": "s",
        "min": "m", "mins": "m",
        "hour": "h", "hours": "h",
        "day": "d", "days": "d",
        "week": "w", "weeks": "w",
        "month": "mo", "months": "mo",
    }
    try:
        return f"{value}{suffixes[unit]}"
    except KeyError as error:
        raise ValueError(f"Temporalidad de IBKR no reconocida: {bar_size}") from error


def _normalizar_fechas(df):
    df = df.copy()
    fechas = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = fechas.dt.tz_localize(None)
    return df.dropna(subset=["date"])

def get_data(
    ib,
    contract,
    mode="update",
    duration="1 M",
    bar_size="1 day",
    use_rth=True,
    pause=1.0,
    request_timeout=15,
    max_requests=None,
):
    if mode not in {"backfill", "update"}:
        raise ValueError("mode debe ser 'backfill' o 'update'")

    ticker = contract.symbol
    timeframe = _timeframe_slug(bar_size)
    file_name = DATA_DIR / f"{ticker}_{timeframe}_history.parquet"

    if file_name.exists():
        print(f"Archivo encontrado: {file_name.name}. Se conservará y actualizará.")
        df_final = _normalizar_fechas(pd.read_parquet(file_name))
        df_final = df_final[BAR_COLUMNS]
    else:
        df_final = pd.DataFrame()

    if mode == "update" and df_final.empty:
        print(f"{ticker}: no existe histórico; se iniciará un backfill.")
        mode = "backfill"

    last_saved = df_final["date"].max() if mode == "update" else None
    if last_saved is not None:
        print(f"{ticker}: actualizando desde {last_saved}.")
    else:
        print(f"{ticker}: iniciando backfill histórico.")

    end_datetime = ""
    request_number = 0
    previous_oldest = None

    while max_requests is None or request_number < max_requests:
        request_number += 1
        print(f"{ticker}: descargando bloque {request_number}...")

        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=use_rth,
            formatDate=2,
            keepUpToDate=False,
            timeout=request_timeout,
        )

        if not bars:
            print(f"{ticker}: IBKR no devolvió más datos.")
            break

        df_block = _normalizar_fechas(util.df(bars))
        if df_block.empty:
            break
        df_block = df_block[BAR_COLUMNS]

        oldest = df_block["date"].min()
        if previous_oldest is not None and oldest >= previous_oldest:
            print(f"{ticker}: la descarga dejó de avanzar; fin del histórico.")
            break
        previous_oldest = oldest

        reached_last_saved = (
            mode == "update" and oldest <= last_saved
        )

        if mode == "update":
            df_block = df_block[df_block["date"] > last_saved]

        if df_block.empty:
            print(f"{ticker}: ya estaba actualizado.")
            break

        df_final = pd.concat([df_final, df_block], ignore_index=True)
        df_final = (
            df_final.drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )

        df_final.to_parquet(file_name, index=False)
        print(
            f"{ticker}: {len(df_block)} velas recibidas; "
            f"histórico desde {df_final['date'].min()} hasta {df_final['date'].max()}."
        )

        if reached_last_saved:
            print(f"{ticker}: actualización completada.")
            break

        end_datetime = (oldest - timedelta(seconds=1)).to_pydatetime()
        ib.sleep(pause)

    print(f"Datos de {ticker} guardados en {file_name} ({len(df_final)} velas).")
    return df_final
