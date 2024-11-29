from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Add lags

def apply_lag(df, lag=1, exclude_columns=None, drop_na=True):
    """
    Aplica un desfase (lag) a todas las columnas de un DataFrame excepto a las especificadas.
    Renombra las columnas desfazadas con el formato "nombre_columna_lagN" y elimina filas con NaN opcionalmente.

    Args:
        df (pd.DataFrame): El DataFrame original.
        lag (int): El número de desfases (lags) a aplicar.
        exclude_columns (list): Lista de nombres de columnas a excluir del desfase.
        drop_na (bool): Si True, elimina las filas con valores NaN después de aplicar el desfase.

    Returns:
        pd.DataFrame: Un DataFrame con las columnas desfazadas y filas opcionalmente limpiadas.
    """
    if exclude_columns is None:
        exclude_columns = []

    # Separar las columnas que no se deben desfazar (Se incluyen aquellas booleanas)
    excluded = df[exclude_columns + list(df.select_dtypes(include=['int64']).columns)]
    
    # Desfazar las columnas restantes
    lagged = df.drop(columns=exclude_columns).shift(lag)
    
    # Renombrar las columnas desfazadas
    lagged.columns = [f"{col}_lag{lag}" for col in lagged.columns]
    
    # Combinar de nuevo las columnas excluidas con las desfazadas
    result = pd.concat([excluded, lagged], axis=1)
    
    # Eliminar filas con valores NaN si drop_na es True
    if drop_na:
        result = result.dropna()
    
    return result

# Scaling values function

def scale_dataframe(df, independent_columns, target_column, feature_range=(0, 1), return_scalers=False):
    """
    Escala las columnas independientes y la columna dependiente de un DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame que contiene los datos.
    - independent_columns (list): Lista de nombres de las columnas independientes (predictoras).
    - target_column (str): Nombre de la columna dependiente (objetivo).
    - feature_range (tuple): Rango de valores para la escala (por defecto (0, 1)).
    - return_scalers (bool): Si True, devuelve también los escaladores usados.

    Returns:
    - X_scaled (pd.DataFrame): DataFrame con las columnas independientes escaladas.
    - y_scaled (pd.DataFrame): Serie o DataFrame con la columna dependiente escalada.
    - (Opcional) scaler_X, scaler_y: Los escaladores utilizados para X y y.
    """
    # Escaladores para las variables independientes y dependiente
    scaler_X = MinMaxScaler(feature_range=feature_range)
    scaler_y = MinMaxScaler(feature_range=feature_range)
    
    # Escalar las columnas independientes
    X_scaled = scaler_X.fit_transform(df[independent_columns])
    X_scaled = pd.DataFrame(X_scaled, columns=independent_columns, index=df.index)
    
    # Escalar la columna dependiente
    y_scaled = scaler_y.fit_transform(df[[target_column]])
    y_scaled = pd.DataFrame(y_scaled, columns=[target_column], index=df.index)
    
    if return_scalers:
        return X_scaled, y_scaled, scaler_X, scaler_y
    return X_scaled, y_scaled

# Generation of train, validation and test sets

def split_data(X_scaled, y_scaled, n_test, n_val):
    """
    Divide los datos en conjuntos de entrenamiento, validación y testeo.

    Parameters:
    - X_scaled (pd.DataFrame): Datos escalados de las variables independientes.
    - y_scaled (pd.DataFrame): Datos escalados de la variable dependiente.
    - n_test (int): Número de días para el conjunto de testeo.
    - n_val (int): Número de días para el conjunto de validación.

    Returns:
    - X_train, y_train: Conjuntos de entrenamiento.
    - X_val, y_val: Conjuntos de validación.
    - X_test, y_test: Conjuntos de testeo.
    """
    # Conjuntos de testeo
    X_test = X_scaled[-n_test:]
    y_test = y_scaled[-n_test:]
    
    # Conjuntos de validación
    X_val = X_scaled[-(n_test + n_val):-n_test]
    y_val = y_scaled[-(n_test + n_val):-n_test]
    
    # Conjuntos de entrenamiento
    X_train = X_scaled[:-(n_test + n_val)]
    y_train = y_scaled[:-(n_test + n_val)]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Train Model with LSTM and apply Cross-Validation

def time_series_lstm_cv(X_scaled, y_scaled, scaler_y, splits, test_size, validation_size, daily_index):
    """
    Realiza un modelo LSTM con validación cruzada tipo TimeSeriesSplit y genera gráficos de pérdida.

    Parámetros:
        X_scaled (pd.DataFrame): Variables predictoras escaladas.
        y_scaled (pd.Series): Variable objetivo escalada.
        scaler_y (Scaler): Scaler usado para escalar y (necesario para desescalar predicciones).
        splits (int): Número de splits para el TimeSeriesSplit.
        test_size (int): Tamaño del conjunto de test.
        validation_size (int): Tamaño del conjunto de validación.
        daily_index (pd.Index): Índice de fechas correspondiente al dataset.

    Retorna:
        historical_predictions (list): Predicciones acumuladas en el conjunto de test.
        historical_real_values (list): Valores reales correspondientes en el conjunto de test.
        cutoff_dates (list): Fechas de inicio de cada conjunto de test.
        mse_per_fold (list): Errores MSE para cada fold.
    """
    tscv = TimeSeriesSplit(n_splits=splits, test_size=test_size)
    historical_predictions = []
    historical_real_values = []
    cutoff_dates = []
    mse_per_fold = []

    for fold, (block_1_indices, test_index) in enumerate(tscv.split(X_scaled)):
        # Índices de entrenamiento, validación y test
        train_index = np.unique(np.abs(block_1_indices - validation_size))
        validation_index = block_1_indices[-validation_size:]

        # Separar los conjuntos
        X_train = X_scaled.iloc[train_index, :]
        X_val = X_scaled.iloc[validation_index, :]
        X_test = X_scaled.iloc[test_index, :]
        y_train = y_scaled.iloc[train_index]
        y_val = y_scaled.iloc[validation_index]
        y_test = y_scaled.iloc[test_index]

        # Reformatear para LSTM
        X_train_lstm = X_train.to_numpy().reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_lstm = X_val.to_numpy().reshape((X_val.shape[0], 1, X_val.shape[1]))
        X_test_lstm = X_test.to_numpy().reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Crear y entrenar el modelo LSTM
        model = Sequential([
            Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
            LSTM(50, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        history = model.fit(
            X_train_lstm,
            y_train,
            epochs=20,
            batch_size=16,
            validation_data=(X_val_lstm, y_val),
            verbose=0
        )

        # Generar gráfico de pérdida para el fold actual
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
        plt.plot(history.history['val_loss'], label='Pérdida de validación')
        plt.title(f'Fold {fold + 1}: Pérdida de entrenamiento y validación')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid()
        plt.show()

        # Realizar predicciones en el conjunto de test
        predictions = model.predict(X_test_lstm)
        predictions_original = scaler_y.inverse_transform(predictions)
        y_test_original = scaler_y.inverse_transform(y_test.to_numpy().reshape(-1, 1))

        # Acumular predicciones y valores reales
        historical_predictions.extend(predictions_original.flatten())
        historical_real_values.extend(y_test_original.flatten())
        cutoff_dates.append(daily_index[test_index[1]])  # Fecha de inicio del test actual

        # Calcular y guardar el MSE del fold
        mse = mean_squared_error(y_test_original, predictions_original)
        mse_per_fold.append(mse)

    return historical_predictions, historical_real_values, cutoff_dates, mse_per_fold

# Graph of the loss function from each fold

def plot_mse_per_fold(mse_per_fold):
    """
    Grafica el MSE de cada fold obtenido en los conjuntos de prueba (test),
    incluyendo una línea horizontal para la media.

    Parámetros:
        mse_per_fold (list): Lista con los valores de MSE para cada fold.

    Retorna:
        None. (Muestra la gráfica)
    """
    # Calcular la media de los valores de MSE
    mse_mean = np.mean(mse_per_fold)

    # Crear una gráfica de barras para los MSE de cada fold
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(mse_per_fold) + 1), mse_per_fold, color='skyblue', edgecolor='black', label='MSE por Fold')
    
    # Agregar una línea horizontal para la media
    plt.axhline(y=mse_mean, color='red', linestyle='--', linewidth=2, label=f'Media MSE: {mse_mean:.2f}')
    
    # Etiquetas y título
    plt.xlabel('Fold')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE por Fold en los Conjuntos de Prueba')
    plt.xticks(range(1, len(mse_per_fold) + 1))
    
    # Agregar una cuadrícula
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Agregar una leyenda
    plt.legend()
    
    # Ajustar el diseño
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()

# Plot predictions of the model

def plot_time_series_with_predictions(historical_real_values, historical_predictions, cutoff_dates, historical_series):
    """
    Grafica una serie de tiempo con valores históricos reales, predicciones y cortes.

    Parámetros:
        historical_real_values (list): Valores reales usados durante la validación.
        historical_predictions (list): Predicciones generadas por el modelo.
        cutoff_dates (list): Fechas donde ocurren los cortes entre los folds.
        historical_series (pd.Series): Serie de tiempo histórica, por ejemplo daily_Monterrey_pm10['PM10'].

    Retorna:
        None. (Muestra una gráfica interactiva de Plotly)
    """
    # Obtener el nombre de la columna de la serie para el eje Y
    column_name = historical_series.name if historical_series.name else "Valor"

    # Crear serie de valores reales históricos
    historical_data = historical_series.iloc[:-len(historical_real_values)]

    # Crear DataFrame para combinar las fechas, valores reales y predicciones
    df_historical = pd.DataFrame({
        'Fecha': list(historical_data.index) + list(historical_series.index[-len(historical_predictions):]),
        'Valor Real': list(historical_data) + list(historical_real_values),
        'Predicciones': [None] * len(historical_data) + list(historical_predictions)
    })
    df_historical.set_index('Fecha', inplace=True)

    # Crear la figura
    fig = go.Figure()

    # Agregar valores reales históricos
    fig.add_trace(go.Scatter(
        x=df_historical.index,
        y=df_historical['Valor Real'],
        name='Valor Real Histórico',
        mode='lines',
        line=dict(color='blue', width=3)
    ))

    # Agregar predicciones
    fig.add_trace(go.Scatter(
        x=df_historical.index,
        y=df_historical['Predicciones'],
        name='Predicciones',
        mode='lines',
        line=dict(color='red', dash='dash', width=3)
    ))

    # Agregar líneas verticales para los cortes
    for i, cutoff in enumerate(cutoff_dates):
        fig.add_trace(go.Scatter(
            x=[cutoff, cutoff],
            y=[df_historical['Valor Real'].min(), df_historical['Valor Real'].max()],
            mode='lines',
            line=dict(color='green', width=2, dash='dot'),
            name='Cortes entre folds' if i == 0 else None,  # Solo añade una vez a la leyenda
            showlegend=(i == 0)  # Solo el primer trazo aparece en la leyenda
        ))

    # Configurar diseño de la gráfica
    fig.update_layout(
        title='Predicciones LSTM con Validación Forward Chaining',
        xaxis_title='Fecha',
        yaxis_title=f"Valor {column_name}",  # Etiqueta del eje Y
        xaxis=dict(rangeslider=dict(visible=True))  # Rango interactivo
    )

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

