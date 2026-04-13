import os
import numpy as np
import tensorflow as tf


#  CONFIGURACIÓN DE REPRODUCIBILIDAD
# ─────────────────────────────────────────────────────────────

def configurar_semillas(semilla=42):
    """
    Configura las semillas aleatorias para garantizar reproducibilidad.
    Esto asegura que los resultados sean los mismos cada vez que se ejecuta."""
    np.random.seed(semilla)
    tf.random.set_seed(semilla)
    os.environ['PYTHONHASHSEED'] = str(semilla)
    print(f" Semillas configuradas: {semilla}")


#  CONSTANTES DEL PROYECTO
# ─────────────────────────────────────────────────────────────

# Rutas
DIRECTORIO_RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resultados')

# Hiperparámetros de los modelos
CONFIG_SUBENTRENADO = {
    'nombre': 'Subentrenado',
    'epochs': 3,
    'batch_size': 32,
    'descripcion': 'Pocas épocas → Alto bias, no aprende'
}

CONFIG_OPTIMO = {
    'nombre': 'Bien Entrenado',
    'epochs': 200,
    'batch_size': 32,
    'patience': 15,
    'descripcion': 'Early Stopping → Equilibrio bias-varianza'
}

CONFIG_SOBREENTRENADO = {
    'nombre': 'Sobreentrenado',
    'epochs': 300,
    'batch_size': 16,
    'descripcion': 'Modelo complejo + muchas épocas → Alta varianza'
}

# Datos
TEST_SIZE = 0.2
RANDOM_STATE = 42


def imprimir_configuracion():
    """Muestra la configuración actual del proyecto."""
    print("\n" + "=" * 55)
    print(" CONFIGURACIÓN DEL PROYECTO")
    print("=" * 55)
    print(f"  Test size:     {TEST_SIZE*100:.0f}%")
    print(f"  Random state:  {RANDOM_STATE}")
    print(f"  Resultados:    {DIRECTORIO_RESULTADOS}")
    print(f"\n  Modelos:")
    for config in [CONFIG_SUBENTRENADO, CONFIG_OPTIMO, CONFIG_SOBREENTRENADO]:
        print(f"    • {config['nombre']}: {config['descripcion']}")
