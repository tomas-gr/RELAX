# Datos sintéticos del brazo

Desde la raíz de RELAX, generar y comprobar el experimento de referencia:

```powershell
python -m planar_arm.explore_data --samples 10000 --seed 42 --output runs/arm_data
python -m unittest discover -s tests -v
```

Se crean `dataset.npz` y `distribution.png`. La figura muestra los extremos por
partición, histogramas angulares y tres configuraciones de entrenamiento. Se usa
Agg para exportar sin depender del backend de ventanas. `--lengths L1 L2` permite
cambiar la geometría; por defecto ambas longitudes son 1. `runs/` está ignorado
por Git. No se requieren dependencias nuevas.

## Generación y reproducibilidad

`planar_arm.data.generate_dataset(n_samples=10000, seed=42, arm=None)` devuelve
un `ArmDataset` con arrays NumPy float64 de forma `(N, 4)`. Las columnas son
`theta1, theta2, x, y`. Cada par angular se muestrea uniformemente en `[-pi, pi)`
y se evalúa con `PlanarArm.forward`. Se conservan radianes y coordenadas originales,
sin normalización estadística ni ruido. El hombro es absoluto y el codo relativo.

Las particiones usan tres generadores locales PCG64, derivados mediante
`SeedSequence(seed).spawn(3)`, sin alterar el RNG global. Se rechazan eventuales
configuraciones exactamente repetidas dentro o entre conjuntos. El tamaño mínimo
es 10. Los tamaños son `floor(0.8*N)`, `floor(0.1*N)` y el resto; para 10000 son
8000/1000/1000. Cambiar N no garantiza conservar los mismos conjuntos anteriores.

Repetir semilla, cantidad, longitudes y versión del código/NumPy reproduce los
arrays. El NPZ guarda las particiones, columnas, semilla, longitudes, versión de
NumPy y algoritmo RNG; para reproducir un experimento también se debe registrar
la revisión del repositorio. `dataset.validate()` verifica unicidad global,
rango angular, finitud y error cartesiano contra la cinemática exacta.

## Uso desde PyTorch

```python
import torch
from torch.utils.data import DataLoader
from planar_arm.data import generate_dataset

data = generate_dataset(10000, seed=42)
train = data.as_torch("train")  # TensorDataset(ángulos, posiciones)
loader = DataLoader(train, batch_size=64, shuffle=True,
                    generator=torch.Generator().manual_seed(42))
angles, positions = next(iter(loader))  # ambos (batch, 2)
validation = data.as_torch("validation", dtype=torch.float64)
```

El adaptador importa Torch únicamente al llamarlo, copia los datos a CPU y utiliza
float32 por defecto. Ese cambio de precisión introduce redondeo, sin cambiar las
unidades. Para comprobar precisión exacta se usan los arrays originales o float64.
No hay una dirección de predicción impuesta: el consumidor decide qué observar.
El orden de DataLoader requiere su propia semilla si usa `shuffle=True`.

Para reutilizar el archivo guardado sin volver a muestrear:

```python
import numpy as np
from planar_arm import PlanarArm
from planar_arm.data import ArmDataset

with np.load("runs/arm_data/dataset.npz", allow_pickle=False) as saved:
    data = ArmDataset(saved["train"], saved["validation"], saved["test"],
                      int(saved["seed"]), PlanarArm(*saved["lengths"]))
train = data.as_torch("train")
```

## Distribución y limitaciones

Los dos ángulos son uniformes e independientes. Los extremos tienen simetría
rotacional en distribución, pero **no son uniformes en área**. El radio cumple
`r² = L1² + L2² + 2*L1*L2*cos(theta2)`; su distribución concentra probabilidad cerca
de las fronteras radiales. Para longitudes iguales el soporte es el disco de radio
2, con concentración espacial especialmente visible cerca del origen y del borde.

La cinemática directa es determinista. La inversa generalmente tiene dos ramas:
una muestra contiene solo la configuración que la generó, aunque otra postura
pueda alcanzar el mismo punto. Muestrear ángulos explora ambas ramas, pero no
construye pares explícitos de soluciones para cada objetivo. Una futura regresión
inversa con error cuadrático puede promediar posturas incompatibles; la evaluación
debe comprobar el error cartesiano y considerar las soluciones analíticas.

No se incluyen deliberadamente casos singulares exactos ni objetivos inalcanzables;
el muestreo continuo tampoco garantiza cobertura de regiones pequeñas. Se necesitan
pruebas específicas para ellos. Los conjuntos separados evalúan interpolación bajo
la misma distribución, no extrapolación, nuevas longitudes ni cambios de dominio.
No hay ruido, dinámica, colisiones ni límites articulares.

Se mantiene el corte angular de la convención original en ±pi. En una etapa posterior
pueden representarse los ángulos con seno/coseno para tratar su periodicidad, teniendo
en cuenta la restricción de norma unidad de cada par y la conversión de vuelta con
`atan2`. Esta versión conserva exclusivamente las cuatro columnas originales.
