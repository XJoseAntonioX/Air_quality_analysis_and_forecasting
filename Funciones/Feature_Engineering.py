import pandas as pd
import plotly.io as pio
import numpy as np
from datetime import date 
import holidays

def add_holiday_feature(df, start_year, end_year, country='Mexico'):
    """
    Agrega una columna `is_holiday` a un DataFrame con base en días festivos de un país,
    y genera un DataFrame con la lista de días festivos.

    Args:
        df (pd.DataFrame): DataFrame con un índice datetime.
        start_year (int): Año inicial para obtener los días festivos.
        end_year (int): Año final para obtener los días festivos.
        country (str): País para obtener los días festivos (por defecto, México).

    Returns:
        tuple: El DataFrame original con la columna `is_holiday` añadida, 
               y un DataFrame con las fechas y nombres de los días festivos.
    """
    # Asegurarse de que el índice del DataFrame sea datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser de tipo DatetimeIndex.")
    
    # Crear el rango de años
    years = range(start_year, end_year + 1)
    
    # Obtener los días festivos del país
    country_holidays = holidays.CountryHoliday(country, years=years)
    
    # Crear un DataFrame con las fechas y nombres de los días festivos
    festivos = [[date, name] for date, name in country_holidays.items()]
    festivos_df = pd.DataFrame(festivos, columns=["Fecha", "Evento"])
    festivos_df["Fecha"] = pd.to_datetime(festivos_df["Fecha"])
    
    # Agregar la columna `is_holiday` al DataFrame original
    df["is_holiday"] = df.index.isin(festivos_df["Fecha"]).astype(int)
    
    return df, festivos_df

def add_weekend_feature(df):
    """
    Agrega una columna `is_weekend` al DataFrame, indicando si una fecha es fin de semana.

    Args:
        df (pd.DataFrame): DataFrame con un índice de tipo datetime.

    Returns:
        pd.DataFrame: DataFrame con la nueva columna `is_weekend`.
    """
    # Asegurarse de que el índice sea de tipo DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser de tipo DatetimeIndex.")
    
    # Crear la columna 'is_weekend' (1 si es sábado o domingo, 0 en caso contrario)
    df["is_weekend"] = df.index.weekday.isin([5, 6]).astype(int)
    
    return df

def add_cyclic_feature(df, start_date, cycle_days, column_name="is_cycle"):
    """
    Agrega una columna que indica si la fecha cumple con un ciclo definido de días.

    Args:
        df (pd.DataFrame): DataFrame con un índice de tipo datetime.
        start_date (str): Fecha inicial del ciclo en formato 'YYYY-MM-DD'.
        cycle_days (int): Factor de temporalidad en días para el ciclo.
        column_name (str): Nombre de la columna a crear (por defecto, "is_cycle").

    Returns:
        pd.DataFrame: DataFrame con la nueva columna.
    """
    # Asegurarse de que el índice es de tipo DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser de tipo DatetimeIndex.")
    
    # Calcular los días desde la fecha inicial
    df["days_from_start"] = (df.index - pd.to_datetime(start_date)).days
    
    # Crear la columna para el ciclo
    df[column_name] = (df["days_from_start"] % cycle_days == 0).astype(int)
    
    # Eliminar la columna auxiliar 'days_from_start'
    df.drop(columns=["days_from_start"], inplace=True)
    
    return df