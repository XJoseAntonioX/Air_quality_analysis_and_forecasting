from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

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

# Generations of train, validation and test sets

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