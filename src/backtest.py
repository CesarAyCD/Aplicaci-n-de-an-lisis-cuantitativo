import numpy as np
import pandas as pd

TRADING_MINUTES_PER_YEAR = 252 * 390

# Convierte arrays en una serie de pandas
def as_series(values, index, name="signal"):
    return pd.Series(values, index=index, name=name).fillna(0).astype(int)

# Ejecuta un backtest con take profit, stop loss y límite por tiempo
def backtest_triple_barrier(df, signal, horizon, pt_mult, sl_mult, cost_bps=1.0):
    data = df.copy()
    signal = as_series(signal, data.index, name="signal")
    signal_values = signal.to_numpy(dtype=np.int8, copy=False)
    closes = data["close"].to_numpy(dtype=float, copy=False)
    highs = data["high"].to_numpy(dtype=float, copy=False)
    lows = data["low"].to_numpy(dtype=float, copy=False)
    log_returns = data["log_return"].to_numpy(dtype=float, copy=False)
    volatilities = data["target_volatility"].to_numpy(dtype=float, copy=False)
    
    position = np.zeros(len(data), dtype=np.int8)
    gross_strategy_log_return = np.zeros(len(data), dtype=float)
    turnover = np.zeros(len(data), dtype=float)

    i = 0
    while i < len(data) - 1:
        if signal_values[i] != 1:
            i += 1
            continue

        # Valida los datos antes de iniciar
        entry_price = closes[i]
        volatility = volatilities[i]
        if (not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(volatility)):
            i += 1
            continue

        # Límite por tiempo
        last_bar = min(i + horizon, len(data) - 1)
        exit_bar = last_bar
        exit_log_return = np.log(closes[exit_bar] / entry_price)
        
        # Cálculo de take profit y stop loss ajustados por volatilidad
        upper_barrier = entry_price * np.exp(pt_mult * volatility)
        lower_barrier = entry_price * np.exp(-sl_mult * volatility)
        
        for candidate in range(i + 1, last_bar + 1):
            profit_taking = highs[candidate] >= upper_barrier
            stop_loss = lows[candidate] <= lower_barrier
            if stop_loss:
                exit_bar = candidate
                exit_log_return = -sl_mult * volatility
                break
            if profit_taking:
                exit_bar = candidate
                exit_log_return = pt_mult * volatility
                break

        position[i + 1:exit_bar + 1] = 1
        gross_strategy_log_return[i + 1:exit_bar + 1] = log_returns[
            i + 1:exit_bar + 1
        ]

        previous_return = gross_strategy_log_return[i + 1:exit_bar].sum()
        gross_strategy_log_return[exit_bar] = exit_log_return - previous_return
        
        turnover[i + 1] += 1
        turnover[exit_bar] += 1
        
        i = exit_bar

    return build_backtest_result(
        data=data,
        signal=signal,
        position=pd.Series(position, index=data.index, name="position"),
        cost_bps=cost_bps,
        turnover=pd.Series(turnover, index=data.index, name="turnover"),
        gross_strategy_log_return=pd.Series(
            gross_strategy_log_return,
            index=data.index,
            name="gross_strategy_log_return"
        )
    )

# Backtest estándar para estrategias simples
def backtest_position(df, signal, cost_bps=1.0, shift_signal=True):
    data = df.copy()
    signal = as_series(signal, data.index, name="signal")

    if shift_signal:
        position = signal.shift(1).fillna(0).astype(int)
    else:
        position = signal.copy()

    position.name = "position"

    return build_backtest_result(
        data=data,
        signal=signal,
        position=position,
        cost_bps=cost_bps
    )

# Construye el dataframe de resultados
def build_backtest_result(data, signal, position, cost_bps, turnover=None, gross_strategy_log_return=None):
    if turnover is None:
        turnover = position.diff().abs().fillna(position.abs())
    else:
        turnover = pd.Series(turnover, index=data.index, name="turnover")
    cost = (cost_bps / 10_000) * turnover

    if gross_strategy_log_return is None:
        gross_strategy_log_return = position * data["log_return"]
    else:
        gross_strategy_log_return = pd.Series(gross_strategy_log_return, index=data.index)
    strategy_log_return = gross_strategy_log_return - cost
    buy_hold_log_return = data["log_return"].copy()

    result = pd.DataFrame(index=data.index)
    result["signal"] = signal
    result["position"] = position
    result["turnover"] = turnover
    result["cost"] = cost
    result["strategy_log_return"] = strategy_log_return
    result["buy_hold_log_return"] = buy_hold_log_return
    result["equity_strategy"] = np.exp(strategy_log_return.cumsum())
    result["equity_buy_hold"] = np.exp(buy_hold_log_return.cumsum())
    result["n_minutes"] = data["n_minutes"]

    return result

# Calcula la pérdida máxima histórica (max drawdown)
def max_drawdown(equity):
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()

# Calcula el factor para anualizar los datos de las tres temporalidades
def annualization_factor(bt):
    n_minutes = float(bt["n_minutes"].median())
    if n_minutes < 1440:
        return TRADING_MINUTES_PER_YEAR / n_minutes
    if n_minutes < 7 * 1440:
        return 252 / (n_minutes / 1440)
    if n_minutes < 28 * 1440:
        return 52 / (n_minutes / (7 * 1440))
    return 12 / (n_minutes / (30 * 1440))

# Calcula las métricas de rendimiento
def return_metrics(strategy_returns, buy_hold_returns, periods_per_year):
    total_return = np.expm1(strategy_returns.sum())
    buy_hold_return = np.expm1(buy_hold_returns.sum())
    annualized_return = np.expm1(strategy_returns.mean() * periods_per_year)
    buy_hold_annualized_return = np.expm1(buy_hold_returns.mean() * periods_per_year)
    return {
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "excess_return": total_return - buy_hold_return,
        "annualized_return": annualized_return,
        "buy_hold_annualized_return": buy_hold_annualized_return,
        "annualized_excess_return": annualized_return - buy_hold_annualized_return
    }

# Calcula Sharpe y Sortino
def risk_metrics(strategy_returns, periods_per_year):
    volatility = strategy_returns.std()
    downside = strategy_returns[strategy_returns < 0].std()
    sharpe = (
        strategy_returns.mean() / volatility * np.sqrt(periods_per_year)
        if volatility > 0 else np.nan
    )
    sortino = (
        strategy_returns.mean() / downside * np.sqrt(periods_per_year)
        if downside > 0 else np.nan
    )
    return {"sharpe": sharpe, "sortino": sortino}

# Genera métricas de trading
def trading_metrics(bt):
    strategy_returns = bt["strategy_log_return"]
    position_change = bt["position"].diff().fillna(bt["position"])
    active_returns = strategy_returns[bt["position"] == 1]
    return {
        "exposure": bt["position"].mean(),
        "n_trades": int((position_change == 1).sum()),
        "n_exits": int((position_change == -1).sum()),
        "n_turnover_events": int((bt["turnover"] > 0).sum()),
        "total_cost": bt["cost"].sum(),
        "hit_rate_active_bars": (
            (active_returns > 0).mean() if len(active_returns) else np.nan
        ),
        "avg_active_bar_return": (
            active_returns.mean() if len(active_returns) else np.nan
        )
    }

# Función principal. Sirve como orquestador del sistema.
def summarize_backtest(bt):
    strategy_returns = bt["strategy_log_return"]
    buy_hold_returns = bt["buy_hold_log_return"]
    periods_per_year = annualization_factor(bt)
    metrics = return_metrics(strategy_returns, buy_hold_returns, periods_per_year)
    metrics.update(risk_metrics(strategy_returns, periods_per_year))
    metrics.update({
        "max_drawdown": max_drawdown(bt["equity_strategy"]),
        "buy_hold_max_drawdown": max_drawdown(bt["equity_buy_hold"]),
    })
    metrics.update(trading_metrics(bt))
    return metrics
