import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

log_path = "/log/axo.core.decorators"  
cols = ["timestamp", "ms", "log_level", "module", "thread", "event", "scope", "method", "success", "service_time"]

df = pd.read_csv(log_path, names=cols)
df = df[df.success == True]  
df["service_time"] = pd.to_numeric(df["service_time"], errors='coerce')
df = df.dropna(subset=["service_time"])

stats = df.groupby("method")["service_time"].agg(["mean", "median", "std"]).reset_index()
stats.columns = ["Algoritmo", "Media", "Mediana", "Desviación estándar"]

output_dir = os.path.dirname(os.path.abspath(__file__))

fig_box = px.box(df, x="method", y="service_time", title="Distribución de tiempos por algoritmo",
                 labels={"method": "Algoritmo", "service_time": "Tiempo de ejecución (segundos)"},
                 color="method", points="outliers")
boxplot_path = os.path.join(output_dir, "tiempos_ejecucion.html")
fig_box.write_html(boxplot_path)
print(f"Gráfica interactiva guardada en: {boxplot_path}")

fig_bar = px.bar(stats, x="Algoritmo", y=["Media", "Mediana", "Desviación estándar"],
                 title="Estadísticos básicos por algoritmo",
                 labels={"value": "Tiempo de ejecución (segundos)", "variable": "Estadístico"},
                 barmode="group")
barplot_path = os.path.join(output_dir, "estadisticos.html")
fig_bar.write_html(barplot_path)
print(f"Gráfica de estadísticos guardada en: {barplot_path}")
