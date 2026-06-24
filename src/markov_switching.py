import warnings

import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

N_REGIMES = 2
MIN_MARKOV_TRAIN_DAYS = 60

# Este fallback prevee que los fallos detengan el modelo (necesario pues hay varias acciones que no convergieron)
def fallback_regime_model(reason, n_regimes=N_REGIMES):
    return {
        "fallback": True,
        "reason": reason,
        "n_regimes": n_regimes,
    }

# Nuevamente, convierte datos intradía a diarios
def to_daily(df):
    data = df.copy()
    daily = (
        data.groupby("session_date")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
    )

    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    daily["daily_log_return"] = np.log(daily["close"] / daily["close"].shift(1))
    daily["daily_range_pct"] = (daily["high"] - daily["low"]) / daily["close"]
    daily["daily_volatility_21"] = (daily["daily_log_return"].rolling(21).std())
    daily["daily_abs_return"] = daily["daily_log_return"].abs()

    return daily.dropna(subset=["daily_log_return"])

# Entrena una regresión con cambio de régimen de Markov usando statsmodels 
def markov_model(daily, train_end, n_regimes = N_REGIMES):
    train_daily = daily.loc[:train_end].copy()
    y_train_daily = train_daily["daily_log_return"]

    # Comprueba las restricciones iniciales
    if len(y_train_daily) < MIN_MARKOV_TRAIN_DAYS:
        return fallback_regime_model(
            f"solo {len(y_train_daily)} sesiones de train; minimo={MIN_MARKOV_TRAIN_DAYS}",
            n_regimes,
        )
    if not np.isfinite(y_train_daily).all() or y_train_daily.std() <= 1e-10:
        return fallback_regime_model("retornos no finitos o sin varianza", n_regimes)

    try:
        with warnings.catch_warnings():
            # Enviaba warnings cada iteración, aunque el modelo convergiera. Esto sólo los elimina de la terminal
            warnings.simplefilter("ignore", ValueWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            model = MarkovRegression(
                endog=y_train_daily,
                k_regimes=n_regimes,
                trend="c",
                switching_variance=True,
            )
            result = model.fit(
                search_reps=30,
                search_iter=20,
                em_iter=10,
                maxiter=1000,
                disp=False,
            )
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        return fallback_regime_model(
            f"ajuste inestable: {type(exc).__name__}: {exc}", n_regimes
        )
    
    # Validación del modelo
    mle_retvals = getattr(result, "mle_retvals", None) or {}
    if not mle_retvals.get("converged", True):
        return fallback_regime_model("el optimizador no convergio", n_regimes)
    if not np.isfinite(np.asarray(result.params)).all():
        return fallback_regime_model("parametros no finitos", n_regimes)
    
    # Mapeo de régimenes
    train_probs = get_probs(result, y_train_daily.index)
    mapping = mapping_vol(daily = train_daily, probs = train_probs)

    regime_model = {
        "result": result,
        "params": np.asanyarray(result.params),
        "n_regimes": n_regimes,
        "mapping": mapping,
        "train_end": train_end
    }

    return regime_model

# Orquestador
def fit_model(df, train_end, n_regimes = N_REGIMES):
    daily = to_daily(df)
    regime_model = markov_model(daily = daily, train_end = train_end, n_regimes = n_regimes)
    return regime_model

# Aplica el Filto de Hamilton. Intenta detectar cuáles régimenes son señales y cuáles ruido. 
def filter_regimes(daily, regime_model):
    n_regimes = regime_model["n_regimes"]
    params = regime_model["params"]
    mapping = regime_model["mapping"]

    model_daily = daily.dropna(subset = ["daily_log_return"]).copy()

    y = model_daily["daily_log_return"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValueWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        model = MarkovRegression(
            endog=y,
            k_regimes=n_regimes,
            trend="c",
            switching_variance=True,
        )
        filtered_result = model.filter(params)
    probs = get_probs(result = filtered_result, index = y.index)

    mapped = pd.DataFrame(index = probs.index)

    for regime, mapped_reg in mapping.items():
        mapped[f"regime_prob_{mapped_reg}"] = probs[regime]

    prob_cols = sorted([col for col in mapped.columns if col.startswith("regime_prob_")])

    mapped["market_regime"] = (
        mapped[prob_cols].idxmax(axis=1).str.extract(r"(\d+)")[0].astype(int)
    )

    mapped = mapped.sort_index()

    return mapped

def get_probs(result, index):
    probs = result.filtered_marginal_probabilities
    probs = pd.DataFrame(probs)
    probs.index = index
    probs.columns = list(range(probs.shape[1]))

    return probs

# Hace un mapeo de las volatilidades estimadas. Garantiza que el régimen 0 sea baja volatilidad.
def mapping_vol(daily, probs):
    temp = daily.loc[probs.index].copy()
    temp["regime"] = probs.idxmax(axis = 1).astype(int)

    vol_by_regime = (temp.groupby("regime")["daily_log_return"].std().sort_values())
    ordered_reg = vol_by_regime.index.tolist()

    regime_mapping = {regime: mapped_regime for mapped_regime, regime in enumerate(ordered_reg)}

    return regime_mapping

# Sincroniza las probabilidades del régimen a las features actuales
# Siempre inserta como información actual el régimen del día anterior
# Esto evita que en las primeras horas se use información futura
def add_regimes(df_int, df_feat, regime_model):

    if regime_model.get("fallback", False):
        features = df_feat.copy()
        features["market_regime_adj"] = 0
        for regime in range(regime_model["n_regimes"]):
            features[f"regime_prob_{regime}_adj"] = float(regime == 0)
        return features

    daily = to_daily(df_int)
    features = df_feat.copy()
    try:
        daily_reg = filter_regimes(daily = daily, regime_model = regime_model)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        features["market_regime_adj"] = 0
        for regime in range(regime_model["n_regimes"]):
            features[f"regime_prob_{regime}_adj"] = float(regime == 0)
        return features

    daily_reg["market_regime_adj"] = daily_reg["market_regime"].shift(1)

    prob_cols = [col for col in daily_reg.columns if col.startswith("regime_prob")]

    for col in prob_cols:
        daily_reg[f"{col}_adj"] = daily_reg[col].shift(1)

    features["session_date"] = features.index.date
    session_dates = pd.to_datetime(features["session_date"])

    features["market_regime_adj"] = session_dates.map(daily_reg["market_regime_adj"])

    features["market_regime_adj"] = (features["market_regime_adj"].fillna(0).astype(int))

    for col in prob_cols:
        adjusted_col = f"{col}_adj"
        features[adjusted_col] = session_dates.map(daily_reg[adjusted_col])

    # Antes de la primera probabilidad disponible usamos un estado neutral.
    probability_columns = [f"{col}_adj" for col in prob_cols]
    features[probability_columns] = features[probability_columns].fillna(0.0)
    first_probability = probability_columns[0]
    no_probability = features[probability_columns].sum(axis=1).eq(0)
    features.loc[no_probability, first_probability] = 1.0

    return features
