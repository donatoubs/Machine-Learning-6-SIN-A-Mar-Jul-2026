"""Proporciona funciones para calcular métricas de rendimiento
y generar reportes comparativos entre modelos."""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluar_modelo(modelo, X_train, y_train, X_test, y_test, nombre):
    """
    Evalúa un modelo calculando MSE, MAE y R² en train y test.
        modelo: Red neuronal entrenada de Keras
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        nombre (str): Nombre descriptivo del escenario """
    y_pred_train = modelo.predict(X_train, verbose=0).flatten()
    y_pred_test = modelo.predict(X_test, verbose=0).flatten()

    resultado = {
        'nombre': nombre,
        'mse_train': mean_squared_error(y_train, y_pred_train),
        'mse_test': mean_squared_error(y_test, y_pred_test),
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test)
    }

    # Mostrar métricas con formato
    print(f"\n  📊 {nombre}:")
    print(f"     {'Métrica':<8} {'Train':>10} {'Test':>10} {'Δ (brecha)':>12}")
    print(f"     {'-'*42}")
    print(f"     {'MSE':<8} {resultado['mse_train']:>10.4f} {resultado['mse_test']:>10.4f} "
          f"{abs(resultado['mse_test'] - resultado['mse_train']):>12.4f}")
    print(f"     {'MAE':<8} {resultado['mae_train']:>10.4f} {resultado['mae_test']:>10.4f} "
          f"{abs(resultado['mae_test'] - resultado['mae_train']):>12.4f}")
    print(f"     {'R²':<8} {resultado['r2_train']:>10.4f} {resultado['r2_test']:>10.4f} "
          f"{abs(resultado['r2_test'] - resultado['r2_train']):>12.4f}")

    return resultado


def generar_reporte(resultados):
    """ Genera un reporte comparativo final de los tres modeloscon diagnósticos y conclusiones."""
    print("\n" + "=" * 60)
    print(" ANÁLISIS COMPARATIVO FINAL")
    print("=" * 60)

    print("""
  🔴 MODELO SUBENTRENADO (Underfitting):
     • Solo 3 épocas → no tuvo tiempo de aprender
     • Error alto en AMBOS conjuntos (train y test)
     • R² bajo → no captura la relación entre features y precio
     • Diagnóstico: ALTO BIAS, BAJA VARIANZA

  🟢 MODELO BIEN ENTRENADO (Good Fit):
     • Early Stopping encontró el punto óptimo
     • Error bajo y SIMILAR en train y test
     • R² alto y consistente → generaliza correctamente
     • Diagnóstico: EQUILIBRIO BIAS-VARIANZA ✓

  🟡 MODELO SOBREENTRENADO (Overfitting):
     • 300 épocas + modelo complejo (7 capas, ~100k parámetros)
     • Error muy bajo en train, pero ALTO en test
     • Gran brecha entre R² de train y test
     • Diagnóstico: BAJO BIAS, ALTA VARIANZA
    """)

    print(" Tabla Resumen:")
    print(f"     {'Modelo':<18} {'MSE Test':>10} {'MAE Test':>10} {'R² Test':>10} {'Brecha R²':>10}")
    print(f"     {'-'*58}")
    for r in resultados:
        brecha = abs(r['r2_train'] - r['r2_test'])
        print(f"     {r['nombre']:<18} {r['mse_test']:>10.4f} {r['mae_test']:>10.4f} "
              f"{r['r2_test']:>10.4f} {brecha:>10.4f}")

    # Identificar el mejor modelo automáticamente
    mejor = max(resultados, key=lambda r: r['r2_test'])
    print(f"\n Mejor modelo: {mejor['nombre']} (R² Test = {mejor['r2_test']:.4f})")

    print("""
  💡 CONCLUSIÓN:
     El modelo Bien Entrenado logra la mejor generalización gracias al
     Early Stopping, que detiene el entrenamiento cuando la pérdida de
     validación deja de mejorar. Esto previene el sobreajuste y encuentra
     automáticamente el número óptimo de épocas para el dataset.

     En el contexto de predicción de precios de casas, un modelo que
     generaliza bien es fundamental para hacer predicciones confiables
     sobre viviendas que no se han visto antes.
    """)
