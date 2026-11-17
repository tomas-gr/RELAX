# PC-02E: convergencia temporal de la inferencia inversa

Se prolongaron exclusivamente tres trayectorias de PC-02, usando el checkpoint
`runs/pc01_seed42/model.pt`, las inicializaciones registradas y la implementación
`PCTrainer` de PC-02: SGD, tasa angular 0.05 y tasa de estados 0.1. La arquitectura,
energía, variables fijas y libres y criterio de convergencia no cambiaron.
El checkpoint coincide por SHA256 con el registrado en PC-02.

La primera corrida de cada caso ejecutó 10.000 pasos. Antes de interpretarla se
compararon **todos** los arrays guardados con PC-02 hasta el paso 3.000, inclusive,
exigiendo igualdad bit a bit. Los tres prefijos coincidieron. Para decidir la
extensión se definió que, si la energía a 10.000 era menor al 95 % de su valor
al paso 5.000 y aún no se cumplía el criterio PC-02, se ejecutaría hasta 30.000.
Los tres casos cumplieron la regla: sus caídas fueron 69.65 %, 36.37 % y 83.78 %.
Las corridas de 30.000 se reiniciaron con las mismas condiciones iniciales y
reprodujeron bit a bit todos los arrays hasta 10.000.

El criterio PC-02 requiere simultáneamente energía <= 1e-6 y norma conjunta de
gradientes <= 1e-4. No se aplica parada anticipada. Todas las métricas se registran
**después** del número indicado de actualizaciones; el paso 0 es el estado inicial.
Se reconstruyeron con `planar_arm.pc02d.diagnose` las distancias y residuos.
La cinemática exacta se usa únicamente para medir, nunca para actualizar estados.

Los archivos están en `runs/pc02e/`: `results.json`, historias completas y
métricas reconstruidas en NPZ, un gráfico por caso y `comparacion.png`. Los
historiales incluyen valores en cada paso, no solo los puntos de control. Los
pesos se verificaron sin cambios y no se escribieron archivos de PC-01/PC-02.

Para reproducir, usar un directorio nuevo:

```powershell
python -m planar_arm.pc02e --output runs/pc02e_repeticion
python -m unittest discover -s tests -v
```

## Puntos de control

En las tablas: ER es error de salida relajada; ED, error de la red en eval; EF,
error físico; R-D, discrepancia relajada frente a eval; D-F, discrepancia eval
frente a física. Son distancias euclídeas. E1 y E2 son errores internos al
cuadrado multiplicados por 1/2; Es es la energía de salida. Las normas internas
son ||z1-mu1|| y ||z2-mu2||. No se suman las distancias para explicar EF porque
son normas de vectores que pueden compensarse.

### target0_start0 – objetivo (1, 1)
| Paso | E total | E1 | E2 | Es | ER | ED | EF | R-D | D-F |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.964585 | 0 | 0 | 0.964585 | 1.38895 | 1.38895 | 1.32978 | 2.08616e-07 | 0.0843796 |
| 100 | 0.0359908 | 0.0297641 | 0.00473379 | 0.00149298 | 0.054644 | 1.30367 | 1.22523 | 1.24918 | 0.0996195 |
| 300 | 0.0284867 | 0.0236755 | 0.00365616 | 0.00115498 | 0.048062 | 1.17655 | 1.06957 | 1.12875 | 0.119865 |
| 1000 | 0.0124664 | 0.0105402 | 0.00147205 | 0.000454125 | 0.0301372 | 0.829186 | 0.691828 | 0.799352 | 0.137588 |
| 3000 | 0.00397461 | 0.00337028 | 0.000472101 | 0.000132234 | 0.0162624 | 0.489855 | 0.440565 | 0.473679 | 0.0816348 |
| 5000 | 0.002742 | 0.00230238 | 0.000343023 | 9.65983e-05 | 0.0138995 | 0.398519 | 0.378605 | 0.384797 | 0.053061 |
| 10000 | 0.000832158 | 0.000695736 | 0.000106287 | 3.0135e-05 | 0.00776338 | 0.216664 | 0.225447 | 0.209005 | 0.0172339 |
| 30000 | 9.6789e-07 | 8.06059e-07 | 1.25867e-07 | 3.59639e-08 | 0.000268194 | 0.00728347 | 0.0348349 | 0.00701847 | 0.0284289 |

| Paso | θ1 | θ2 | ||grad q|| | ||grad z1|| | ||grad z2|| | ||z1−mu1|| | ||z2−mu2|| | Fuera | ¿Convergió? |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 1.72132 | -0.384038 | 0 | 0 | 2.41378 | 2.89375e-07 | 3.40741e-07 | no | no |
| 100 | 1.66758 | -0.41806 | 0.0102769 | 0.0149201 | 0.0118828 | 0.243984 | 0.0973014 | no | no |
| 300 | 1.57935 | -0.456223 | 0.00899141 | 0.0130875 | 0.0111643 | 0.217603 | 0.0855121 | no | no |
| 1000 | 1.35626 | -0.56706 | 0.00541519 | 0.00798159 | 0.0078052 | 0.145191 | 0.0542596 | no | no |
| 3000 | 1.18618 | -0.76896 | 0.00162162 | 0.00208646 | 0.00166234 | 0.0821009 | 0.0307278 | no | no |
| 5000 | 1.23274 | -0.91956 | 0.00157379 | 0.00181136 | 0.000870082 | 0.0678584 | 0.0261925 | no | no |
| 10000 | 1.39159 | -1.21948 | 0.00107388 | 0.0012246 | 0.000491569 | 0.0373024 | 0.0145799 | no | no |
| 30000 | 1.5593 | -1.52616 | 4.18579e-05 | 4.79434e-05 | 1.77752e-05 | 0.00126967 | 0.000501779 | no | sí |

El mínimo físico observado fue **0.0348349** en el paso 30000; el error de la última iteración fue **0.0348349**. El mínimo es descriptivo y no selecciona ni sustituye la solución final.

### target0_start2 – objetivo (1, 1)
| Paso | E total | E1 | E2 | Es | ER | ED | EF | R-D | D-F |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.06121 | 0 | 0 | 1.06121 | 1.45685 | 1.45685 | 1.46252 | 1.04308e-07 | 0.101784 |
| 100 | 0.0597855 | 0.0458693 | 0.0102154 | 0.00370085 | 0.0860332 | 1.43145 | 1.44528 | 1.34542 | 0.115844 |
| 300 | 0.0577517 | 0.0442427 | 0.009918 | 0.00359103 | 0.084747 | 1.40098 | 1.42657 | 1.31625 | 0.137805 |
| 1000 | 0.0539803 | 0.0411575 | 0.00941086 | 0.00341192 | 0.0826066 | 1.34087 | 1.40735 | 1.25831 | 0.206472 |
| 3000 | 0.0490355 | 0.037173 | 0.00868255 | 0.00317997 | 0.0797492 | 1.2599 | 1.42479 | 1.18028 | 0.357211 |
| 5000 | 0.042532 | 0.0321582 | 0.00756778 | 0.00280604 | 0.0749138 | 1.16146 | 1.41845 | 1.08672 | 0.49127 |
| 10000 | 0.0270643 | 0.020509 | 0.00476186 | 0.00179338 | 0.0598895 | 0.91682 | 1.42173 | 0.857235 | 0.742579 |
| 30000 | 0.0153884 | 0.0118078 | 0.00260752 | 0.000973181 | 0.0441176 | 0.700291 | 1.78506 | 0.656327 | 1.16271 |

| Paso | θ1 | θ2 | ||grad q|| | ||grad z1|| | ||grad z2|| | ||z1−mu1|| | ||z2−mu2|| | Fuera | ¿Convergió? |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | -2.54986 | 2.98842 | 0 | 0 | 2.5596 | 4.43529e-07 | 2.13879e-07 | no | no |
| 100 | -2.52866 | 3.02512 | 0.00628636 | 0.00862341 | 0.00505439 | 0.302884 | 0.142936 | no | no |
| 300 | -2.5164 | 3.08236 | 0.00545092 | 0.00735261 | 0.00413102 | 0.297465 | 0.14084 | no | no |
| 1000 | -2.50969 | 3.23817 | 0.00378162 | 0.00470007 | 0.0020898 | 0.286906 | 0.137192 | sí | no |
| 3000 | -2.67553 | 3.54741 | 0.00385583 | 0.00431567 | 0.000931961 | 0.272665 | 0.131777 | sí | no |
| 5000 | -2.96131 | 3.8611 | 0.00451227 | 0.00504714 | 0.000926498 | 0.253607 | 0.123027 | sí | no |
| 10000 | -3.6185 | 4.66128 | 0.00338749 | 0.00380536 | 0.000571143 | 0.202529 | 0.0975896 | sí | no |
| 30000 | -4.37144 | 6.13662 | 0.000941604 | 0.00108958 | 0.00015128 | 0.153673 | 0.0722153 | sí | no |

El mínimo físico observado fue **1.40358** en el paso 7624; el error de la última iteración fue **1.78506**. El mínimo es descriptivo y no selecciona ni sustituye la solución final.

### target3_start3 – objetivo (-1, 1)
| Paso | E total | E1 | E2 | Es | ER | ED | EF | R-D | D-F |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0488687 | 0 | 0 | 0.0488687 | 0.31263 | 0.31263 | 0.29588 | 3.37175e-07 | 0.166386 |
| 100 | 0.00198217 | 0.00159952 | 0.000294084 | 8.85683e-05 | 0.0133093 | 0.302169 | 0.281375 | 0.289056 | 0.163608 |
| 300 | 0.00170483 | 0.0013817 | 0.000248102 | 7.5032e-05 | 0.0122501 | 0.281804 | 0.252133 | 0.269713 | 0.158014 |
| 1000 | 0.00104727 | 0.0008594 | 0.000143725 | 4.415e-05 | 0.00939681 | 0.224528 | 0.169934 | 0.215197 | 0.142748 |
| 3000 | 0.000361298 | 0.000300883 | 4.58667e-05 | 1.45479e-05 | 0.00539405 | 0.134033 | 0.0497393 | 0.128639 | 0.121963 |
| 5000 | 0.000160602 | 0.000133593 | 2.04692e-05 | 6.5399e-06 | 0.0036166 | 0.0889816 | 0.0304459 | 0.0853716 | 0.11468 |
| 10000 | 2.60505e-05 | 2.15965e-05 | 3.37554e-06 | 1.07838e-06 | 0.00146859 | 0.0356091 | 0.0732949 | 0.034146 | 0.10889 |
| 30000 | 2.76927e-08 | 2.29255e-08 | 3.61453e-09 | 1.15262e-09 | 4.80129e-05 | 0.00115905 | 0.10473 | 0.00111123 | 0.105889 |

| Paso | θ1 | θ2 | ||grad q|| | ||grad z1|| | ||grad z2|| | ||z1−mu1|| | ||z2−mu2|| | Fuera | ¿Convergió? |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 1.64079 | 1.79739 | 0 | 0 | 0.566532 | 3.45886e-07 | 6.19329e-07 | no | no |
| 100 | 1.63607 | 1.7873 | 0.00227691 | 0.00328211 | 0.00134455 | 0.0565601 | 0.0242523 | no | no |
| 300 | 1.62584 | 1.76795 | 0.00209413 | 0.00300012 | 0.00120401 | 0.0525681 | 0.0222756 | no | no |
| 1000 | 1.60136 | 1.70973 | 0.00155156 | 0.00216468 | 0.000950889 | 0.0414584 | 0.0169544 | no | no |
| 3000 | 1.5873 | 1.60148 | 0.000787564 | 0.000989229 | 0.00056133 | 0.0245309 | 0.00957772 | no | no |
| 5000 | 1.60124 | 1.54064 | 0.000499248 | 0.00058751 | 0.000378291 | 0.0163459 | 0.00639838 | no | no |
| 10000 | 1.63471 | 1.46754 | 0.000196447 | 0.000224491 | 0.000150464 | 0.00657219 | 0.00259821 | no | no |
| 30000 | 1.6598 | 1.4194 | 6.78712e-06 | 6.77496e-06 | 5.42328e-06 | 0.000214124 | 8.49745e-05 | no | sí |

El mínimo físico observado fue **0.0276354** en el paso 4391; el error de la última iteración fue **0.10473**. El mínimo es descriptivo y no selecciona ni sustituye la solución final.

## Interpretación de las trayectorias

**target0_start0.** La energía cayó de 0.003975 en 3.000 a 9.68e-7 en
30.000. Los residuos internos y la discrepancia relajada–eval disminuyeron; el
error físico bajó de 0.44056 a 0.03483. El criterio de convergencia se cumplió
por primera vez en el paso 29.910. Permaneció en el dominio entrenado. En este
caso 3.000 pasos fueron claramente insuficientes y ampliar el presupuesto mejoró
las tres métricas.

**target0_start2.** La energía cayó de 0.04904 en 3.000 a 0.01539 en 30.000,
pero no cumplió el criterio de convergencia. La discrepancia relajada–eval bajó
sin desaparecer, de 1.18028 a 0.65633; los residuos internos también decrecieron.
El error físico aumentó de 1.42479 a 1.78506, mientras la discrepancia entre red
y física creció de 0.35721 a 1.16271. El mejor error físico fue 1.40358 en el
paso 7.624. Salió del dominio en el paso 537 y permaneció fuera hasta el final;
los ángulos siguieron cambiando. Esto es evidencia de que menor energía no
implica mayor precisión física en esta trayectoria. El resultado es compatible
con extrapolación deficiente, pero no demuestra que sea la única causa.

**target3_start3.** La energía bajó de 0.000361 en 3.000 a 2.77e-8 en 30.000;
se cumplió el criterio numérico por primera vez en 19.425. La discrepancia
relajada–eval cayó de 0.12864 a 0.00111 y los residuos internos casi desaparecieron.
El error físico fue 0.04974 en 3.000, alcanzó un mínimo de 0.02764 en 4.391 y
terminó en 0.10473. El error de la red directa frente a la física era 0.10589
al final: el acercamiento a una solución consistente con la red no aseguró una
posición física más precisa. Permaneció dentro del dominio entrenado.

En los tres casos no se observaron aumentos de energía total mayores a 1e-7
entre pasos consecutivos. Esa tolerancia no excluye oscilaciones menores o
cambios de dirección de las variables. El segundo caso presenta disminución
lenta hacia 30.000 y no converge; este presupuesto finito no demuestra un
estancamiento permanente. El movimiento angular entre 5.000 y 10.000 fue 0.3394,
1.0355 y 0.0804 rad respectivamente, por lo que las variables seguían cambiando.

## Conclusiones y próximos pasos

Verificado: 3.000 pasos eran insuficientes para la convergencia numérica de los
tres ejemplos. Más pasos mejoraron energía y consistencia, pero el efecto físico
fue distinto: mejoró en el primer caso, empeoró en el segundo y pasó por un mínimo
antes de empeorar en el tercero. Dos casos cumplieron el criterio PC-02 y uno no.
Una energía baja mide el modelo aprendido, no garantiza exactitud física.

Hipótesis: la extrapolación de la red en el segundo caso y el error de aproximación
observado en el tercero contribuyen al comportamiento; estas observaciones no
prueban causalidad ni garantizan que más iteraciones no puedan cambiarlo.
Como siguiente experimento conviene estudiar tasas de aprendizaje mediante una
grilla pequeña predefinida, manteniendo estos casos y reportando costo, consistencia,
dominio y error físico. También conviene estudiar el dominio angular en una etapa
separada. Continuar indefinidamente el presupuesto no se justifica por estos datos:
dos casos ya convergieron y el tercero presenta deterioro físico, aunque un mayor
presupuesto podría aclarar su comportamiento asintótico.

## Verificación ejecutada

Se ejecutaron las 30 pruebas de `unittest` con el intérprete `relax`; todas
pasaron. Las nuevas cubren rechazo de prefijos distintos, regla de extensión,
criterio de convergencia y separación de error físico. Las comparaciones bit a
bit de prefijos se ejecutaron para las tres corridas reales. No se repitió el
entrenamiento ni se cambiaron los pesos. La comparación de fuentes incluye las
trayectorias previas y el SHA256 del checkpoint en `results.json`.
