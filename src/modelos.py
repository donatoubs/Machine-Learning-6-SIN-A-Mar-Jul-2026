import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibilidad
tf.random.set_seed(42)


def crear_modelo_simple(n_features):
    """Red neuronal simple para regresión de precios.
    Arquitectura: Input(n) → 32(ReLU) → 16(ReLU) → 1

    n_features: Número de features de entrada (8 para California Housing)"""
    modelo = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(n_features,),
                     name='capa_oculta_1'),
        layers.Dense(16, activation='relu', name='capa_oculta_2'),
        layers.Dense(1, name='salida')
    ], name='modelo_simple')

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return modelo


def crear_modelo_complejo(n_features):
    """Red neuronal compleja para demostrar overfitting.
    Arquitectura: Input(n) → 64 → 64 → 32 → 32 → 1
    Sin regularización para facilitar la memorización.

    n_features: Número de features de entrada """
    modelo = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(n_features,),
                     name='capa_oculta_1'),
        layers.Dense(64, activation='relu', name='capa_oculta_2'),
        layers.Dense(32, activation='relu', name='capa_oculta_3'),
        layers.Dense(32, activation='relu', name='capa_oculta_4'),
        layers.Dense(1, name='salida')
    ], name='modelo_complejo')

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )

    return modelo

#  ESCENARIO 1: MODELO SUBENTRENADO (Underfitting)
# ─────────────────────────────────────────────────────────────

def entrenar_modelo_subentrenado(X_train, y_train, X_test, y_test):
    """MODELO SUBENTRENADO (Underfitting):
     Arquitectura simple (2 capas ocultas)
     Solo 3 épocas de entrenamiento
     No alcanza a aprender los patrones del dataset

    Resultado esperado:
        Alto error en train Y en test
        R² bajo en ambos conjuntos
        Diagnóstico: ALTO BIAS"""
    print("\n" + "=" * 60)
    print("  🔴 MODELO SUBENTRENADO (Underfitting)")
    print("     Épocas: 3 | Arquitectura: Simple")
    print("=" * 60)

    n_features = X_train.shape[1]
    modelo = crear_modelo_simple(n_features)

    historia = modelo.fit(
        X_train, y_train,
        epochs=3,               # MUY POCAS épocas
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    return modelo, historia

#  ESCENARIO 2: MODELO BIEN ENTRENADO (Good Fit)
# ─────────────────────────────────────────────────────────────

def entrenar_modelo_optimo(X_train, y_train, X_test, y_test):
    """MODELO BIEN ENTRENADO (Good Fit):
     Arquitectura simple (2 capas ocultas)
     200 épocas con Early Stopping (patience=15)
     Restaura los mejores pesos automáticamente
     Encuentra el punto óptimo de entrenamiento

    Resultado esperado:
        Error bajo y SIMILAR en train y test
        R² alto y consistente
        Diagnóstico: EQUILIBRIO BIAS-VARIANZA """
    print("\n" + "=" * 60)
    print("  🟢 MODELO BIEN ENTRENADO (Good Fit)")
    print("     Épocas: 200 + Early Stopping | Arquitectura: Simple")
    print("=" * 60)

    n_features = X_train.shape[1]
    modelo = crear_modelo_simple(n_features)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    historia = modelo.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    return modelo, historia

#  ESCENARIO 3: MODELO SOBREENTRENADO (Overfitting)
# ─────────────────────────────────────────────────────────────

def entrenar_modelo_sobreentrenado(X_train, y_train, X_test, y_test):
    """MODELO SOBREENTRENADO (Overfitting):
     Arquitectura COMPLEJA (7 capas ocultas, ~100k parámetros)
     100 épocas sin Early Stopping
     Sin regularización (sin Dropout, sin L2)

    Resultado esperado:
         Error MUY bajo en train pero ALTO en test
         Gran brecha entre R² de train y R² de test
         Diagnóstico: ALTA VARIANZA """
    print("\n" + "=" * 60)
    print("  🟡 MODELO SOBREENTRENADO (Overfitting)")
    print("     Épocas: 100 | Arquitectura: Compleja | Sin regularización")
    print("=" * 60)

    n_features = X_train.shape[1]
    modelo = crear_modelo_complejo(n_features)

    historia = modelo.fit(
        X_train, y_train,
        epochs=50,              # Muchas épocas + modelo complejo = memoriza
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1
    )

    print(f"  Entrenamiento completado: {len(historia.history['loss'])} épocas")

    return modelo, historia

