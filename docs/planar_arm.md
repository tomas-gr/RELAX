# Brazo planar 2R

Desde la raíz de RELAX, con el entorno `relax` activo:

```powershell
python -m planar_arm
python -m planar_arm --angles 0.4 -0.8 --target 1 1 --lengths 1 1
python -m planar_arm --target 3 0
python -m planar_arm --target 0 0
python -m planar_arm --save runs/arm.png
python -m unittest discover -s tests -v
```

Para guardar una figura, la carpeta de destino debe existir. `--save` utiliza el backend Agg y no necesita un sistema de ventanas. La apertura
interactiva depende de que el backend gráfico del entorno esté operativo; en la
validación local Tk no encontró `init.tcl`, por lo que se validó la exportación PNG.

El hombro está fijo en el origen. `theta1` es absoluto y `theta2` es relativo al
primer segmento; todos los ángulos están en radianes. Las longitudes son positivas,
configurables y usan la misma unidad que las coordenadas. No hay límites articulares.

`PlanarArm.forward` devuelve el extremo; `joint_positions` devuelve hombro, codo y
extremo. `inverse` devuelve `IKResult`: cero soluciones para objetivos inalcanzables,
dos para puntos regulares y una cuando las ramas coinciden en una singularidad.
Los ángulos se normalizan a `[-pi, pi)`. La primera rama tiene seno del codo positivo,
la segunda negativo (salvo coincidencia singular).

El espacio alcanzable es el anillo de radios `abs(L1-L2)` y `L1+L2`. Con longitudes
iguales, el origen admite cualquier ángulo de hombro con el codo plegado: se informa
`infinite_solutions=True` y se devuelve `(0, -pi)` como representante. Una tolerancia
relativa al alcance máximo, por defecto `1e-10`, absorbe pequeñas discrepancias de
redondeo fuera de las fronteras. Esos puntos se resuelven sobre la frontera, por lo
que el residuo puede ser de ese orden. Cerca de una singularidad la inversa es
sensible a pequeñas perturbaciones del objetivo.

La visualización muestra la configuración directa y cada solución inversa en
paneles con la misma escala espacial. El objetivo aparece como una cruz roja;
los ángulos actuales y el extremo se indican en cada panel.

Los módulos matemático y gráfico son independientes. La cinemática exacta, los
pares sintéticos que permite generar y la visualización podrán servir como
referencia física y evaluación del futuro modelo PC. Esta etapa no incluye redes,
entrenamiento ni dinámica de inferencia.
