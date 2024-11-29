import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss 

# Seasonality and time series graphics

def graficar_serie_tiempo(df, contaminante, columnas_resaltadas):
    """
    Graficar una serie de tiempo con puntos resaltados basados en columnas específicas del DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con la serie de tiempo en el índice y datos a graficar.
    contaminante : str
        Nombre de la columna principal a graficar.
    columnas_resaltadas : list of str
        Lista de nombres de columnas binarias (0 o 1) para resaltar puntos.
    """
    fig = go.Figure()

    # Graficar la serie principal
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[contaminante],
        name=contaminante,
        mode='lines',
        line=dict(color='blue')
    ))

    # Resaltar puntos para cada columna en `columnas_resaltadas`
    for col in columnas_resaltadas:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame.")
        
        # Filtrar los puntos donde la columna tiene valor 1
        puntos_resaltados = df[df[col] == 1]
        fig.add_trace(go.Scatter(
            x=puntos_resaltados.index,
            y=puntos_resaltados[contaminante],
            name=col,
            mode='markers',
            marker=dict(size=8, symbol='circle')
        ))

    # Configurar el rango deslizante y los botones de rango
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    # Mostrar la gráfica
    fig.show()

def Periodograma(ts, detrend='linear', window='boxcar', scaling='density', ts_frequency=None, show_minor_ticks=True, axsize=(12, 3)):
    """
    Take timeseris and plot the periodogram(s).
    If data has multiple columns, one plot is made
    for each column.
    The time period format (df.index.to_period()) is
    preferred for clarity, but any format will work.

    Parameters
    ----------
    ts : DataFrame or Series with data to visualize
    detrend : whether and how to detrend the data
    window : the shape of the time window
    scaling : return either power spectral density
              or power spectrum
    ts_frequency : override timeseries index frequency
                   (for display only, in the xlabel)
    show_minor_ticks : plot minor ticks
    axsize : the size of a single plot in the figure

    Returns
    -------
    filtered_periods_spectra: List of tuples (filtered_periods, filtered_spectrum) for each time series
    """

    filtered_periods_spectra = []
    max_period = 182.5  # Half a year in days

    if not isinstance(ts, pd.Series) and not isinstance(ts, pd.DataFrame):
        raise Exception(f'data must be pd.Series or pd.DataFrame')
    if isinstance(ts, pd.Series):
        ts = pd.DataFrame(ts)

    num_cols = len(ts.columns.to_list())
    fig, ax = plt.subplots(num_cols, 1, figsize=(axsize[0], num_cols * axsize[1]), layout='constrained')
    if num_cols == 1:
        ax = np.array([ax])

    for i in range(num_cols):
        c = ts.columns.to_list()[i]
        frequencies, spectrum = periodogram(
            ts[c],
            fs=1,
            detrend=detrend,
            window=window,
            scaling=scaling,
        )
        with np.errstate(divide='ignore'):
            periods = 1 / frequencies

            # Apply filter to exclude periods greater than 182.5 days
            mask = periods <= max_period
            filtered_periods = periods[mask]
            filtered_spectrum = spectrum[mask]
            filtered_periods_spectra.append((filtered_periods, filtered_spectrum))

        # Plot filtered data
        ax[i].step(filtered_periods, filtered_spectrum, color="purple")
        ax[i].set_xscale("log")
        ax[i].xaxis.set_major_formatter('{x:.0f}')
        if show_minor_ticks:
            ax[i].xaxis.set_minor_formatter('{x:.0f}')
            ax[i].grid(visible=True, which='both', axis='both')
        else:
            ax[i].grid(visible=True, which='major', axis='both')

        xlabel = f'Periodo (Días)'
        if ts_frequency is not None:
            xlabel += f', freqstr: {ts_frequency}'
        else:
            if hasattr(ts.index, 'freqstr'):
                xlabel += f', freqstr: {ts.index.freqstr}'
        ax[i].set_xlabel(xlabel)
        if scaling == 'density':
            ax[i].set_ylabel("power spectral density")
        else:
            ax[i].set_ylabel("power spectrum")
        ax[i].set_title(c)

    fig.suptitle('Filtered Periodogram')
    fig.show()

    return filtered_periods_spectra

def Top_10_Periodogram(df, column_name, list_periods_spectra):
    """
    Muestra los 10 principales periodos y sus espectros para la columna especificada del DataFrame.
    
    Parámetros:
        df (pd.DataFrame): El DataFrame con los datos.
        column_name (str): El nombre de la columna a analizar.
        list_periods_spectra (list): Lista de tuplas con periodos y espectros correspondientes.
    """
    # Verificar si la columna existe en el DataFrame
    if column_name not in df.columns:
        raise ValueError(f"La columna '{column_name}' no se encuentra en el DataFrame proporcionado.")

    # Extraer la columna como un pandas Series
    series = df[column_name]
    
    # Nombre de la columna (variable)
    variable_name = series.name if series.name is not None else column_name

    # Iterar sobre la lista de periodos y espectros
    for i, (periods, spectrum) in enumerate(list_periods_spectra):
        # Ordenar espectros en orden descendente y tomar los top 10
        top_indices = np.argsort(spectrum)[-10:][::-1]
        top_periods = periods[top_indices]
        top_spectra = spectrum[top_indices]
        
        # Crear DataFrame para la serie actual
        df_top = pd.DataFrame({
            "top_periods": top_periods,
            "top_spectra": top_spectra
        }).sort_values(by="top_spectra", ascending=False)

        # Mostrar el nombre de la variable
        print(f"\nTop 10 para la variable '{variable_name}':")
        display(df_top)

# Correlations

def heat(df, variable, lags_max):
    """
    Genera mapas de calor que muestran las correlaciones entre una variable específica
    y las variables independientes con diferentes retrasos (lags),
    utilizando los métodos Pearson, Spearman y Kendall.
    
    Parámetros:
        data (pd.DataFrame): El DataFrame con los datos.
        variable (str): La variable que no será transformada en lag.
        lags_max (int): El número máximo de lags a considerar.
    """

    # Variables independientes
    variables_independientes = list(df.columns)
    variables_independientes.remove(variable)

    # Configurar la figura para subplots
    nrows = lags_max  # Una fila por cada lag
    ncols = 3         # Tres columnas: Pearson, Spearman, Kendall
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4 * lags_max))
    fig.suptitle(f"Correlaciones con '{variable}' por Lags y Métodos", fontsize=16, y=0.95)

    # Iterar sobre cada lag y calcular correlaciones
    for lag in range(1, lags_max + 1):
        lagged_df = pd.DataFrame()

        # Generar los lags para las variables independientes
        for col in variables_independientes:
            lagged_df[f"{col}_lag{lag}"] = df[col].shift(lag)

        lagged_df[variable] = df[variable]

        # Eliminar filas con valores NaN generados por los lags
        lagged_df = lagged_df.dropna()

        # Calcular las correlaciones para cada método
        methods = ['pearson', 'spearman', 'kendall']
        for col_idx, method in enumerate(methods):
            corr_with_variable = lagged_df.corr(method=method)[[variable]].drop(index=variable)

            # Graficar el mapa de calor
            ax = axes[lag - 1, col_idx]
            sns.heatmap(
                corr_with_variable,
                annot=True,
                cmap='coolwarm',
                fmt=".2f",
                cbar=True,
                vmin=-1,
                vmax=1,
                ax=ax
            )
            ax.set_title(f"Lag {lag} - {method.capitalize()} Correlation", fontsize=12)
            ax.set_ylabel("Variables Idependientes" if col_idx == 0 else "")
            ax.set_xlabel("Variable Dependiente" if lag == lags_max else "")
            # Centrar el tick de la variable en el eje x
            ax.set_xticks([0.5])  # Posición centrada
            ax.set_xticklabels([variable])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

# Stationarity Tests

def prueba_adf(serie):
    print("-- ADF a nivel --")
    resultado = adfuller(serie, regression = "c")
    print(f'Estadístico ADF: {resultado[0]}')
    print(f'Valor p: {resultado[1]}')
    print('Valores Críticos:')
    if resultado[1] <= 0.05:
        print("La serie SI es estacionaria a nivel (rechazamos la hipótesis nula).")
    else:
        print("La serie NO es estacionaria a nivel (no rechazamos la hipótesis nula).")
        
    print("\n-- ADF de tendencia --")
    resultado = adfuller(serie, regression = "ct")
    print(f'Estadístico ADF: {resultado[0]}')
    print(f'Valor p: {resultado[1]}')
    print('Valores Críticos:')
    if resultado[1] <= 0.05:
        print("La serie SI es estacionaria a nivel y tendencia (rechazamos la hipótesis nula).")
    else:
        print("La serie NO es estacionaria a nivel y tendencia (no rechazamos la hipótesis nula).")

def prueba_kpss(serie):
    print("-- KPSS a nivel --")
    resultado = kpss(serie, regression = "c")
    print(f'Estadístico KPSS: {resultado[0]}')
    print(f'Valor p: {resultado[1]}')
    print('Valores Críticos:')
    if resultado[1] <= 0.05:
        print("La serie NO es estacionaria a nivel (rechazamos la hipótesis nula).")
    else:
        print("La serie SI es estacionaria a nivel (no rechazamos la hipótesis nula).")
    
    print("\n-- KPSS de tendencia --")
    resultado = kpss(serie, regression = "ct")
    print(f'Estadístico KPSS: {resultado[0]}')
    print(f'Valor p: {resultado[1]}')
    print('Valores Críticos:')
    if resultado[1] <= 0.05:
        print("La serie NO es estacionaria a tendencia (rechazamos la hipótesis nula).")
    else:
        print("La serie SI es estacionaria a tendencia (no rechazamos la hipótesis nula).")