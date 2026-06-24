[README.md](https://github.com/user-attachments/files/29277070/README.md)
# Plataforma de backtesting y auditoría de estrategias financieras

Este proyecto implementa una plataforma de investigación en Python para evaluar señales de trading, modelos de clasificación y reglas de backtesting sobre datos históricos de IBKR.

El objetivo no es presentar un robot listo para operar en vivo, sino construir una herramienta reproducible para analizar estrategias financieras, comparar modelos, medir riesgo, detectar sobreajuste y evaluar la diferencia entre validación y test.

## Descripción general

El sistema carga datos históricos por activo y temporalidad, genera variables financieras, entrena modelos de machine learning, selecciona umbrales usando exclusivamente el conjunto de validación y evalúa el desempeño final en test.

El flujo general es:

- Carga de datos históricos de IBKR.
- Limpieza, ordenamiento y validación de columnas.
- Generación de features técnicos, contexto de mercado y régimen.
- Construcción de etiquetas mediante triple-barrier.
- División en train, validation y test.
- Entrenamiento de modelos especializados por serie y modelos generalistas.
- Selección de thresholds en validation.
- Backtesting con costos, exposición, drawdown y retorno.
- Exportación de resultados a Parquet.
- Visualización en dashboard.

## Estructura del proyecto

project/
    data/
    outputs/
    src/
        backtest.py
        baselines.py
        carga_datos.py
        context_features.py
        feature_eng.py
        markov_switching.py
        model_metrics.py
        plots.py
        result_reporting.py
        results_io.py
        run_backtest.py
        train_models.py
    streamlit_app.py
    requirements.txt
    README.md

## Requisitos

Se recomienda usar Python 3.10 o superior.

Instalación de dependencias:

pip install -r requirements.txt

## Datos de entrada

Los datos deben estar en la carpeta data con nombres del tipo:

AAPL_15m_history.parquet
AAPL_1h_history.parquet
AAPL_1d_history.parquet

Cada archivo debe contener, como mínimo, las siguientes columnas:

date, open, high, low, close, volume, average, barCount

IB_main los genera por defecto con este formado, por lo que no debería ser un problema. El uso de parquet en vez de csv se debe principalmente a que al usar tantos datos csv podría ralentizar el proceso, además que el peso del output podría ser excesivo. Por ejemplo, el parquet con los datos de las curvas pesa casi 6GB estando ya en este formato que lo comprime.

El módulo de carga valida estas columnas, elimina duplicados por fecha, ordena y añade datos como ticker o duración de la barra.

## Feature engineering

El proyecto genera diferentes grupos de variables:

- Variables de sesión: barra dentro de la sesión, día de la semana, apertura y cierre.
- Variables de precio: retornos logarítmicos, rangos, volatilidad, ATR y posición del cierre dentro de la vela.
- Variables de volumen y liquidez: volumen relativo, z-score de volumen, barCount, precio promedio reportado por IBKR y volumen promedio por transacción.
- Indicadores técnicos: medias móviles, RSI, Bollinger Bands y distancia a medias.
- Contexto de mercado: SPY, QQQ, ETF sectorial, retornos relativos, beta rolling y correlación con SPY.
- Régimen de mercado: modelo Markov-Switching ajustado sobre datos diarios del periodo de entrenamiento.

## Target triple-barrier

La etiqueta principal se construye con una lógica tipo triple-barrier:

- Barrera superior: profit-taking.
- Barrera inferior: stop-loss.
- Barrera vertical: horizonte máximo de barras.

Uutiliza high y low para detectar toques de barrera dentro de cada vela. Esto hace que el backtest sea más realista que usar únicamente precios de cierre. Esto podría optimizarse todavía más añadiendo velas de 1 minuto.

Las columnas close, high y low se conservan para simular el backtest, pero se excluyen del entrenamiento de modelos para reducir riesgo de leakage.

## Modelos

Se entrenan cinco familias de modelos:

- Logistic Regression.
- Random Forest.
- Extra Trees.
- HistGradientBoosting.
- XGBoost.

Para cada serie se entrenan modelos especializados por ticker y timeframe. Además, se entrenan modelos generalistas usando únicamente los datos de train de todas las series disponibles. Este último tiene por objetivo ver si el modelo logra captar señales del mercado gracias a aumentar la cantidad de inputs.

## Separación temporal

Cada serie se divide cronológicamente en:

60% train
20% validation
20% test

También se aplica una purga de barras en las fronteras para reducir el riesgo de contaminación entre splits por el horizonte del target.

El flujo de uso de datos es:

- Train: ajuste de modelos.
- Validation: selección de umbrales y ranking de modelos.
- Test: evaluación final fuera de muestra.

## Backtesting

El backtest evalúa estrategias long/cash.

Cuando una señal se activa, la estrategia entra en posición long. La salida se determina mediante take profit, stop loss o vencimiento del horizonte máximo.

Se descuentan costos de transacción. En la configuración base se utiliza COST_BPS = 1.0.

Las métricas principales son:

- Retorno total.
- Retorno anualizado.
- Retorno excedente contra Buy & Hold.
- Sharpe.
- Sortino.
- Máximo drawdown.
- Exposición.
- Número de trades.
- Costos totales.

## Baselines

El proyecto incluye reglas base para comparar los modelos contra estrategias simples:

- Buy & Hold.
- Momentum.
- Reversión a la media.
- Trend following.
- RSI.
- Bollinger.
- Breakout con volumen.

Estas señales se usan como benchmarks y también como features binarias dentro de los modelos.

## Ejecución del backtest

Si no se tienen datos:

python IB_main.py

Si se desean distintos tickers debe editarse el archivo

Para ejecutar el proyecto: 

python src/run_backtest.py


El script genera los archivos en `outputs/`.

## Dashboard

Se puede ejecutar con:

streamlit run app/streamlit_app.py


El dashboard permite explorar resultados por ticker, timeframe, split, estrategia y modelo. También muestra curvas de equity, drawdowns, thresholds, métricas de clasificación, importancia de variables y reportes validation vs test.

## Resultados principales

El experimento final evalúa múltiples activos y temporalidades. Los resultados muestran que muchas señales que parecieran buenas en validation pierden fuerza en test.

La conclusión principal es que los modelos no generan alpha robusto frente a Buy & Hold en la mayoría de los casos. Sin embargo, sí permiten analizar exposición, reducción de drawdown, estabilidad de señales y degradación fuera de muestra.

Por tanto, el valor del proyecto está en el análisis de estrategias y no en afirmar que existe una estrategia lista para operar.

## Limitaciones

- No se modela spread variable, slippage real ni impacto de mercado.
- Los hiperparámetros de los modelos están fijados manualmente. Se intentó cambiar esto pero no era posible con el tamaño del modelo, y se optó por usarlo así en vez de simplificar todo.
- Las señales direccionales presentan métricas de clasificación cercanas al azar.
- La selección en validation no es comparable con su rendimiento en test.
- El proyecto no contempla ejecución en vivo.

## Mejoras futuras

Algunas extensiones posibles son:

- Reglas de selección más conservadoras.
- El uso de datos en live para optimizar parámetros del take profit y stop loss.
- Análisis separado por timeframe.
- Incorporación de restricciones operativas más realistas.

## Nota final

Este proyecto tiene fines académicos y de investigación. No funciona ni pretende ser asesoría financiera, recomendación de inversión ni sistema de trading.

El modelo presentado en el vídeo fue entrenado con:

TICKER_GROUPS = {
    "broad_market": (
        "SPY", "QQQ", "IWM", "DIA", "VTI", "MDY",
    ),
    "sectors": (
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "XLB", "XLRE", "XLC",
    ),
    "macro": (
        "TLT", "IEF", "SHY", "HYG", "LQD", "GLD", "SLV",
        "USO", "VNQ", "EEM", 
        "EFA", "ARKK",
    ),
    "technology": (
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
        "TSLA", "AMD", "INTC", "CRM", "ORCL", "IBM", "NFLX",
    ),
    "cyclical": (
        "JPM", "BAC", "GS", "XOM", "CVX", "CAT", "BA", "GE",
    ),
    "defensive_consumer": (
        "JNJ", "UNH", "LLY", "PFE", "WMT", "COST", "HD",
        "MCD", "KO", "PEP", "DIS",
    ),
}

Por limitaciones de Git, no se incluyen los inputs (son 500MB) ni outputs (son 6GB). Se añaden sólo unos pocos datos como ejemplo para hacer más fácil la ejecución del proyecto.
