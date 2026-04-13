"""
Dataset: Precios de viviendas en distritos de California.

Features (8):
    - MedInc:       Ingreso medio del distrito
    - HouseAge:     Antigüedad media de las casas
    - AveRooms:     Número medio de habitaciones
    - AveBedrms:    Número medio de dormitorios
    - Population:   Población del distrito
    - AveOccup:     Ocupación media por vivienda
    - Latitude:     Latitud del distrito
    - Longitude:    Longitud del distrito
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def cargar_datos():
    #Carga el California Housing Dataset desde scikit-learn.
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names

    print("=" * 55)
    print("  DATASET: California Housing")
    print("=" * 55)
    print(f"  Muestras totales: {X.shape[0]:,}")
    print(f"  Features:         {X.shape[1]}")
    print(f"  Target:           Precio medio de vivienda ($100k)")
    print(f"\n  Features disponibles:")
    for i, name in enumerate(feature_names):
        print(f"    {i+1}. {name}")

    return X, y, feature_names


def explorar_datos(X, y, feature_names):
    """
    Realiza un análisis exploratorio del dataset:
    estadísticas descriptivas, distribución del target y correlaciones.

        X: Features del dataset
        y: Target (precios)
        feature_names: Nombres de las columnas
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['MedHouseVal'] = y

    print("\n" + "=" * 55)
    print("  EXPLORACIÓN DE DATOS")
    print("=" * 55)

    print("\n  Estadísticas descriptivas:")
    print(df.describe().round(2).to_string())

    print(f"\n  Rango de precios: ${y.min()*100:.0f}k — ${y.max()*100:.0f}k")
    print(f"  Precio medio:     ${y.mean()*100:.0f}k")
    print(f"  Precio mediano:   ${np.median(y)*100:.0f}k")

    print(f"\n  Correlación de cada feature con el precio:")
    correlaciones = df.corr()['MedHouseVal'].drop('MedHouseVal').sort_values(ascending=False)
    for feat, corr in correlaciones.items():
        barra = "█" * int(abs(corr) * 20)
        signo = "+" if corr > 0 else "-"
        print(f"    {feat:<15} {signo}{abs(corr):.4f}  {barra}")

    return df


def preparar_datos(X, y, test_size=0.2):
    """
    Divide los datos en train/test y aplica estandarización.
        X: Features
        y: Target
        test_size: Proporción para test (default: 20%)
    """
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Estandarización (media=0, std=1) — esencial para redes neuronales
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n Datos de entrenamiento: {len(X_train):,} muestras ({int((1-test_size)*100)}%)")
    print(f"  Datos de prueba:        {len(X_test):,} muestras ({int(test_size*100)}%)")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    X, y, names = cargar_datos()
    df = explorar_datos(X, y, names)
    preparar_datos(X, y)
    print("\n Módulo de datos funciona correctamente")
