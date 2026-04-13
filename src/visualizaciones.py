"""Genera:
    1. Curvas de aprendizaje (loss por época)
    2. Predicciones vs valores reales (scatter)
    3. Comparación de métricas (barras)
    4. Distribución de errores (histogramas) """

import os
import numpy as np
import matplotlib.pyplot as plt

# Directorio de resultados
RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resultados')
os.makedirs(RESULTADOS, exist_ok=True)


def graficar_curvas_aprendizaje(historias):
    """Grafica las curvas de aprendizaje (train loss vs val loss) lado a lado.
    Permite observar underfitting, good fit y overfitting. """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colores = ['#e74c3c', '#2ecc71', '#f39c12']
    titulos = [
        '🔴 Subentrenado\n(3 épocas)',
        '🟢 Bien Entrenado\n(Early Stopping)',
        '🟡 Sobreentrenado\n(300 épocas)'
    ]

    for i, (historia, titulo) in enumerate(zip(historias, titulos)):
        ax = axes[i]
        epochs = range(1, len(historia.history['loss']) + 1)

        ax.plot(epochs, historia.history['loss'],
                label='Train Loss', color=colores[i], linewidth=2)
        ax.plot(epochs, historia.history['val_loss'],
                label='Validation Loss', color=colores[i],
                linewidth=2, linestyle='--', alpha=0.8)

        ax.set_title(titulo, fontsize=13, fontweight='bold')
        ax.set_xlabel('Época')
        ax.set_ylabel('MSE (Loss)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Curvas de Aprendizaje — Comparación de Modelos',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    ruta = os.path.join(RESULTADOS, 'curvas_aprendizaje.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Guardada: {ruta}")


def graficar_predicciones(modelos, X_test, y_test, resultados):
    """ Scatter plot: predicciones vs valores reales para cada modelo.
    La línea diagonal perfecta indica predicciones exactas."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colores = ['#e74c3c', '#2ecc71', '#f39c12']
    titulos = ['🔴 Subentrenado', '🟢 Bien Entrenado', '🟡 Sobreentrenado']

    for i, (modelo, resultado) in enumerate(zip(modelos, resultados)):
        ax = axes[i]
        y_pred = modelo.predict(X_test, verbose=0).flatten()

        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color=colores[i])

        # Línea de predicción perfecta
        lim_min = min(y_test.min(), y_pred.min())
        lim_max = max(y_test.max(), y_pred.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                'k--', linewidth=1.5, alpha=0.5, label='Predicción perfecta')

        ax.set_title(f'{titulos[i]}\nR²={resultado["r2_test"]:.4f}',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Precio Real ($100k)')
        ax.set_ylabel('Precio Predicho ($100k)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Predicciones vs Precios Reales',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    ruta = os.path.join(RESULTADOS, 'predicciones_vs_reales.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Guardada: {ruta}")


def graficar_comparacion_metricas(resultados):
    #Barras comparando MSE, MAE y R² de train vs test para los 3 modelos.
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    etiquetas = ['Subentrenado', 'Óptimo', 'Sobreentrenado']
    colores_train = ['#e74c3c88', '#2ecc7188', '#f39c1288']
    colores_test = ['#e74c3c', '#2ecc71', '#f39c12']
    x = np.arange(len(etiquetas))
    ancho = 0.3

    metricas = [
        ('MSE', 'mse_train', 'mse_test'),
        ('MAE', 'mae_train', 'mae_test'),
        ('R²', 'r2_train', 'r2_test')
    ]

    for idx, (titulo, key_tr, key_te) in enumerate(metricas):
        vals_tr = [r[key_tr] for r in resultados]
        vals_te = [r[key_te] for r in resultados]

        for j in range(3):
            axes[idx].bar(x[j] - ancho/2, vals_tr[j], ancho,
                          color=colores_train[j],
                          label='Train' if j == 0 else '')
            axes[idx].bar(x[j] + ancho/2, vals_te[j], ancho,
                          color=colores_test[j], edgecolor='black',
                          linewidth=0.5, label='Test' if j == 0 else '')

        axes[idx].set_title(f'{titulo} por Modelo', fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(etiquetas, fontsize=10)
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)

    plt.suptitle('Comparación de Métricas entre Modelos',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    ruta = os.path.join(RESULTADOS, 'comparacion_metricas.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f" Guardada: {ruta}")


def graficar_distribucion_errores(modelos, X_test, y_test):
    """ Histogramas de la distribución de errores de predicción.
    Un modelo bien entrenado tiene errores centrados en 0."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colores = ['#e74c3c', '#2ecc71', '#f39c12']
    titulos = ['🔴 Subentrenado', '🟢 Bien Entrenado', '🟡 Sobreentrenado']

    for i, modelo in enumerate(modelos):
        ax = axes[i]
        y_pred = modelo.predict(X_test, verbose=0).flatten()
        errores = y_test - y_pred

        ax.hist(errores, bins=50, color=colores[i], alpha=0.7, edgecolor='black',
                linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(x=errores.mean(), color=colores[i], linestyle='-',
                   linewidth=2, label=f'Media: {errores.mean():.3f}')

        ax.set_title(f'{titulos[i]}\nStd: {errores.std():.3f}',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Error de Predicción ($100k)')
        ax.set_ylabel('Frecuencia')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Distribución de Errores de Predicción',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    ruta = os.path.join(RESULTADOS, 'distribucion_errores.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print(f" Guardada: {ruta}")
