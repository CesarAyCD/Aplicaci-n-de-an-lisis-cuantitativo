from ib_async import *
from data import get_data

ib = IB()

try:
    ib.connect('127.0.0.1', 4002, clientId=2)
    
    if ib.isConnected():
        print("Conexión a IB establecida con éxito.")
    else:
        print("No se pudo conectar.")

except Exception as e:
    print(f"Error al intentar conectar: {e}")

googl = Stock('QQQ', 'SMART', 'USD')
BAR_SIZE = "1 day"  # Se usan: "X mins", "X hour", "X day", "X week"
DURATION = "1 M"

ib.qualifyContracts(googl)

df_google = get_data(ib, googl, mode="update", bar_size=BAR_SIZE)

print(df_google.head())
