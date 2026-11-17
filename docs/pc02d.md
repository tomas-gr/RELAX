# PC-02D: diagnóstico de inferencia inversa

Se analizaron las 32 trayectorias guardadas de PC-02: cuatro objetivos, cuatro
inicializaciones y dos métodos. No se reentrenó PC-01 ni se ajustaron tasas. El
checkpoint se verificó contra el SHA256 registrado en PC-02. Los archivos
anteriores se leyeron sin modificarlos. El análisis completo está en
`runs/pc02d/diagnostics.json` y un NPZ por trayectoria en `runs/pc02d/`.

Para reproducir, mover o renombrar previamente la salida existente y ejecutar:

```powershell
python -m planar_arm.pc02d
python -m unittest discover -s tests -v
```

La salida rechaza directorios no vacíos. Los gráficos de casos representativos son
`runs/pc02d/target0_start0.png`, `target0_start2.png` y `target3_start3.png`.
Los datos en `runs/` están ignorados por Git. Las trayectorias originales guardan
estados y residuos PC, pero no predicciones eval ni componentes de gradiente: se
reconstruyeron a partir de los ángulos, estados y pesos registrados. Las posiciones
físicas provienen de `PlanarArm.forward` registradas en PC-02.

## Definiciones y descomposición

Sea `y*` el objetivo, `yR` la salida con estados relajados, `f(q)` la pasada
directa eval y `p(q)` la posición física. Se registran por iteración las normas
`||yR-y*||`, `||f(q)-y*||`, `||p(q)-y*||`, `||f(q)-p(q)||` y `||yR-f(q)||`.
Las dos últimas describen diferencias de modelo y de estados; sus normas no son
contribuciones aditivas del error físico, pues las direcciones pueden compensarse.
También se guarda `||z1-mu1||`, `||z2-mu2||` y los tres términos
`E1=0.5||z1-mu1||²`, `E2=0.5||z2-mu2||²`, `Eout=0.5||yR-y*||²`.

| Caso final PC | Error relajado | Error eval | Error físico | Modelo vs físico | Relajado vs eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| (1,1), inicio 0 | 0.01626 | 0.48986 | 0.44056 | 0.08163 | 0.47368 |
| (1,1), inicio 2 | 0.07975 | 1.25990 | 1.42479 | 0.35721 | 1.18028 |
| (-1,1), inicio 3 | 0.00539 | 0.13403 | 0.04974 | 0.12196 | 0.12864 |

En el primer caso, la brecha relajado–eval es mucho mayor que el error del modelo
contra la física: hay fuerte inconsistencia entre estados y ángulos, además de
error de aproximación. El tercer caso muestra que errores vectoriales pueden
compensarse: el error físico es menor que cualquiera de las dos discrepancias.
En el segundo aparecen ambas fuentes y los ángulos salen del dominio entrenado.
No se puede asignar una proporción causal exacta mediante estas normas.

## Energía, gradientes y pasos

En (1,1), inicio 0, la energía inicial fue aproximadamente `(E1,E2,Eout) =
(0,0,0.964585)`; al finalizar fue `(0.003370, 0.000472, 0.000132)`. La energía
total bajó a 0.003975, pero los errores internos permanecieron. En las 16
trayectorias PC no hubo aumentos de energía reconstruida superiores a 1e-7 entre
pasos consecutivos; esto no descarta oscilaciones de variables o aumentos menores.
El caso (1,1), inicio 2, finalizó con `(0.037173,0.008683,0.003180)`.

Los gradientes reconstruidos usan `r1=z1-mu1`, `r2=z2-mu2` y `ro=yR-y*`:

```text
gq  = -W1ᵀ r1
gz1 = r1 - diag(1-tanh(z1)²) W2ᵀ r2
gz2 = r2 + diag(1-tanh(z2)²) W3ᵀ ro
```

En el caso (1,1), inicio 0, las normas `(q,z1,z2)` fueron aproximadamente
`(0,0,2.414)` al inicio, `(0,0.604,1.462)` tras un paso y `(0.0186,0.499,1.019)`
tras dos pasos. Esto concuerda con la transmisión del error desde la salida hacia
z2, z1 y finalmente q; los valores iniciales reconstruidos de orden 1e-7 reflejan
el redondeo al recalcular en float64 a partir de datos float32.

Al final, las normas de gradiente fueron `(0.001622,0.002087,0.001662)` y las
normas del último paso `(0.0000811,0.0002087,0.0001662)`. Son pequeñas respecto
a los gradientes iniciales, pero no prueban convergencia. Con SGD sin momentum,
el callback multiplica `gq` por `0.05/0.1`, y el optimizador aplica 0.1 al grupo:
el paso efectivo angular es `-0.05*gq`, y los de estados `-0.1*gz`. La diferencia
máxima por componente entre pasos observados y reconstruidos fue 3.08e-7 en las
16 trayectorias, compatible con redondeo float32.

La máxima discrepancia de energía reconstruida frente a los registros fue
1.90e-7; la de normas de gradiente, 1.01e-6. En el checkpoint real se compararon
gradientes vectoriales con autograd en pasos 0, 1, 2, 1000 y 3000 del primer
caso: diferencia máxima 1.4e-15 en float64. Diferencias finitas centrales con
paso 1e-5 verificaron una componente de cada variable en esos mismos pasos, con
error máximo 9.7e-12. La prueba automática repite el contraste con otro modelo.
Esto comprueba las derivadas de la energía implementada y la adaptación de SGD,
no la corrección física de la solución.

Como indicador descriptivo de saturación se contó `|tanh(z)| > 0.99`: el máximo
en cualquier instante y prueba fue 5/32 neuronas en la primera capa y 2/32 en la
segunda. No se demuestra que la saturación sea la causa dominante. Tres de las
16 trayectorias PC salieron alguna vez de `[-pi,pi)`; (1,1), inicio 2, pasó 2464
de 3001 estados registrados fuera. La red no garantiza periodicidad ahí.

## Energías de referencia física

Para cada solución analítica se calculó `f(q)` y se inicializaron conceptualmente
los estados a las predicciones de una pasada directa: E1=E2=0 y
`E=0.5||f(q)-y*||²`. Son energías **sin relajar** los estados; no son mínimos
condicionados de energía. Las dos ramas pueden tener errores de modelo distintos.

| Objetivo | Energía rama 1 | Energía rama 2 |
| --- | ---: | ---: |
| (1,1) | 0.000765 | 0.000492 |
| (1,0) | 0.001301 | 0.000714 |
| (0,1) | 0.012669 | 0.001219 |
| (-1,1) | 0.006608 | 0.044600 |

Para (1,1), inicio 0, ambas energías de referencia son menores que la energía
final 0.003975 de PC. Existe por tanto una configuración físicamente correcta
con menor energía evaluada, pero eso no demuestra que la dinámica llegue a ella.
En (-1,1), inicio 3, la energía final PC 0.000361 es menor que las referencias
físicas sin relajar, pese a un error físico de 0.04974: optimizar la energía
aprendida permite estados y ángulos distintos de una solución física exacta.

## Interpretación y próximos experimentos

**Verificado:** la salida relajada puede ser engañosamente precisa con estados
inconsistentes; el modelo también tiene error físico; hay pasos fuera del dominio;
los gradientes y pasos reconstruidos coinciden con la implementación. Ninguna de
las 16 pruebas PC cumplió el criterio de convergencia de PC-02.

**Hipótesis:** la dinámica de relajación conjunta puede avanzar lentamente por
diferencias de escala o curvatura, y la extrapolación angular puede crear mínimos
incorrectos. Estos datos no establecen una causa única, ni descartan mínimos
locales o problemas derivados del entrenamiento del modelo.

Prioridad sugerida: (1) ampliar el presupuesto para unos pocos casos fijos y
seguir todos los errores y gradientes; (2) comparar tasas de estados y ángulos en
una grilla pequeña predefinida, con los mismos inicios y referencia convencional;
(3) estudiar una restricción al dominio y discontinuidades angulares, reportando
por separado cualquier cambio de objetivo; (4) evaluar mejoras del modelo directo
con validación independiente antes de repetir IK. Ninguna propuesta se ejecutó en
este diagnóstico.

Se ejecutaron 27 pruebas automáticas con `relax`, todas exitosas. La comprobación
del checkpoint real y las diferencias finitas se ejecutaron adicionalmente como
análisis puntual. No se repitieron las inferencias completas ni el entrenamiento.
