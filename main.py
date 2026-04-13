"""Este script demuestra tres escenarios de entrenamiento de redes neuronales
para predecir el precio medio de viviendas en California:

    1. SUBENTRENADO (Underfitting)  → 3 épocas, no aprende los patrones
    2. BIEN ENTRENADO (Good Fit)    → Early Stopping, generaliza bien
    3. SOBREENTRENADO (Overfitting) → 300 épocas, modelo complejo, memoriza ruido """

import numpy as np

# Módulos propios
from src.utils import configurar_semillas, imprimir_configuracion
from src.datos import cargar_datos, explorar_datos, preparar_datos
from src.modelos import (
    entrenar_modelo_subentrenado,
    entrenar_modelo_optimo,
    entrenar_modelo_sobreentrenado
)
from src.evaluacion import evaluar_modelo, generar_reporte
from src.visualizaciones import (
    graficar_curvas_aprendizaje,
    graficar_predicciones,
    graficar_comparacion_metricas,
    graficar_distribucion_errores
)

def main():
    """Pipeline principal de ejecución."""

    print("=" * 60)
    print("  PREDICCIÓN DE PRECIOS DE CASAS")
    print("  CON REDES NEURONALES")
    print("  Dataset: California Housing")
    print("=" * 60)

    # Configurar reproducibilidad
    configurar_semillas()
    imprimir_configuracion()


    # PASO 1: Cargar y explorar el dataset
    print("\n PASO 1: Cargando dataset...")
    X, y, feature_names = cargar_datos()
    df = explorar_datos(X, y, feature_names)

    # PASO 2: Preparar datos (split + escalar)
    print("\n PASO 2: Preparando datos...")
    X_train, X_test, y_train, y_test, scaler = preparar_datos(X, y)

    # PASO 3: Entrenar los tres modelos
    print("\n PASO 3: Entrenando modelos...")

    modelo_sub, hist_sub = entrenar_modelo_subentrenado(
        X_train, y_train, X_test, y_test
    )
    modelo_opt, hist_opt = entrenar_modelo_optimo(
        X_train, y_train, X_test, y_test
    )
    modelo_sobre, hist_sobre = entrenar_modelo_sobreentrenado(
        X_train, y_train, X_test, y_test
    )

    modelos = [modelo_sub, modelo_opt, modelo_sobre]
    historias = [hist_sub, hist_opt, hist_sobre]

    # PASO 4: Evaluar modelos
    print("\n" + "=" * 60)
    print(" PASO 4: Evaluando modelos...")
    print("=" * 60)

    res_sub = evaluar_modelo(modelo_sub, X_train, y_train, X_test, y_test, "Subentrenado")
    res_opt = evaluar_modelo(modelo_opt, X_train, y_train, X_test, y_test, "Bien Entrenado")
    res_sobre = evaluar_modelo(modelo_sobre, X_train, y_train, X_test, y_test, "Sobreentrenado")

    resultados = [res_sub, res_opt, res_sobre]

    # PASO 5: Generar visualizaciones
    print(" PASO 5: Generando gráficas...")

    graficar_curvas_aprendizaje(historias)
    graficar_predicciones(modelos, X_test, y_test, resultados)
    graficar_comparacion_metricas(resultados)
    graficar_distribucion_errores(modelos, X_test, y_test)

    # PASO 6: Análisis final
    generar_reporte(resultados)

    print("=" * 60)
    print(" EJECUCIÓN COMPLETADA")
    print(" Gráficas guardadas en la carpeta resultados/")
    print("=" * 60)


if __name__ == "__main__":
    main()
