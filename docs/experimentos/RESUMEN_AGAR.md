# YOLOv8m y YOLOv8m + MLCA sobre el subdataset oficial de AGAR

## 1. Qué cambia respecto a la versión anterior

La versión anterior comparó YOLOv8m con YOLOv8m + MLCA sobre el subdataset v5 propio (102 fotos de entrenamiento) y concluyó que el MLCA no aportaba una mejora distinguible del ruido entre seeds. En esta versión se mantiene el modelo, la integración del MLCA y los hiperparámetros, y se cambia únicamente **el dataset**: ahora es un subdataset del **AGAR oficial**.

| | **Versión anterior** | **Esta versión (AGAR)** |
|---|---|---|
| dataset | subdataset v5 propio | subdataset del AGAR oficial |
| fotos de train / valid / test | 102 / 24 / 31 | **5823 / 1344 / 1792** |
| colonias por foto | ~10 (máx. 45) | **~26** (train: 149 469 cajas; valid: 35 401; test: 49 292) |
| clases | 3 | 3 (*E. coli*, *P. aeruginosa*, *S. aureus*) |
| modelos | YOLOv8m y YOLOv8m + MLCA | los mismos |
| hiperparámetros | los de `best_v8.pt` | los mismos (Adam, lr 1e-4, 100 épocas, patience 15, imgsz 640, batch 32) |
| duración de una corrida (T4, suma de sesiones) | ~13 min | **4 h 08** (base) y **5 h 03** (MLCA) |

El cambio de escala es de casi dos órdenes de magnitud en cantidad de fotos de entrenamiento. Por eso la comparación sobre AGAR responde una pregunta distinta de la anterior: si el MLCA aporta cuando **no** faltan datos.

## 2. El dataset

- **Origen:** AGAR oficial. Se tomó únicamente lo correspondiente a las tres especies del proyecto: *E. coli*, *P. aeruginosa* y *S. aureus*.
- **Publicación:** proyecto `agar_subdataset`, versión 1, en Roboflow (formato YOLOv8).
- **Clases:** el notebook las lee del `data.yaml` descargado, de modo que los nombres siempre coinciden con los identificadores de las etiquetas.
- **Especie de la placa:** los nombres de archivo de AGAR son numéricos y no contienen la especie. La especie verdadera de cada placa se obtiene de sus etiquetas (clase mayoritaria de las cajas anotadas).

## 3. Diseño del experimento

Se entrenan dos variantes con exactamente la misma configuración, de modo que la única diferencia sea el MLCA:

- **`base`:** YOLOv8m preentrenado en COCO, sin modificaciones (control).
- **`mlca`:** YOLOv8m con los 4 bloques C2f del neck reemplazados por C2f-MLCA (8 módulos MLCA). El backbone no cambia.

**Métricas** (sobre valid y test completos, con el `best.pt` de cada corrida, elegido por mAP50-95 en valid):

- mAP50 y mAP50-95 (`model.val`).
- Especie por placa: voto mayoritario de las cajas detectadas (conf 0,5; NMS 0,45).
- Error de conteo: error absoluto medio entre colonias detectadas y anotadas por foto.

**Seeds:** se ejecutó con `SEEDS = [0]` (una corrida por variante, ~9 h en total). Con una sola seed no se puede separar del todo el efecto del MLCA del ruido entre corridas (ver sección 7).

**Verificación del dataset:** al descargarlo, el notebook confirmó 5823 fotos en train, 1344 en valid y 1792 en test, tal como figuran en Roboflow (149 469 / 35 401 / 49 292 cajas).

## 4. Entrenamiento en Colab: guardado y reanudación

En la primera ejecución las corridas `base_0` (2 h 47 min) y `mlca_0` (3 h) se copiaban a Drive recién **al terminar**; al desconectarse Colab se perdieron los resultados. El notebook se corrigió así:

- Cada `SYNC_EVERY` épocas (por defecto 2) se copia a Drive la carpeta completa de la corrida: `last.pt`, `best.pt`, `results.csv` y las gráficas. Cada archivo se copia con nombre temporal y se renombra, para evitar archivos corruptos si la desconexión ocurre durante la copia.
- Si Colab se desconecta, al volver a ejecutar las secciones 1 a 8 la corrida interrumpida **se retoma desde la última época guardada** (`resume=True`) y las terminadas se saltean.
- La sección 10 muestra las curvas de entrenamiento leyendo desde Drive (también de corridas a medias) y genera `curvas_comparacion.png`.
- La sección 11 arma un `.zip` por corrida con `best.pt`, `results.csv`, `args.yaml` y todas las gráficas, y lo descarga al equipo.
- La evaluación guarda `resultados_colab.csv` después de cada corrida y no repite las ya evaluadas.

**Funcionó en la ejecución final:** el `results.csv` de ambas corridas muestra que el contador de tiempo se reinició dos veces (`base_s0` en las épocas 13 y 49; `mlca_s0` tras las épocas 13 y 52). Es decir, cada corrida se retomó dos veces y se completó sin perder avance. La duración real de cada corrida es la suma de sus tres sesiones: ≈ 248 min (`base_s0`, 76 épocas) y ≈ 303 min (`mlca_s0`, 74 épocas).

Nota técnica: al retomar, ultralytics reinicia el contador de *early stopping*, por lo que una corrida retomada puede entrenar más épocas que una ininterrumpida. No afecta a `best.pt`, que conserva la mejor época global (`base_s0`: época 42; `mlca_s0`: época 59).

Opcionalmente, `VAL_SUBSET` permite validar cada época con un subconjunto fijo de valid (por ejemplo 400 fotos), lo que reduce el tiempo por corrida en unos 25 minutos. La evaluación final siempre usa valid y test completos.

## 5. Corrección de la evaluación por especie

La primera evaluación de `base_s0` dio una especie por placa de 0,5 % (valid) y 0,7 % (test). Ese valor **no es un resultado del modelo**: la función obtenía la especie verdadera desde el nombre del archivo y, al ser numérico, siempre resultaba vacía. Se corrigió tomando la especie de las etiquetas de cada placa. Todas las demás métricas no se veían afectadas.

## 6. Resultados finales (seed 0)

Evaluación con el `best.pt` de cada corrida sobre valid y test completos.

| | valid `base_s0` | valid `mlca_s0` | test `base_s0` | test `mlca_s0` |
|---|---|---|---|---|
| mAP50 | 0,961 | 0,963 | 0,958 | 0,955 |
| mAP50-95 | 0,638 | 0,638 | 0,620 | 0,617 |
| especie por placa | 97,0 % | 96,8 % | 96,8 % | 97,0 % |
| error de conteo (colonias por foto) | 1,685 | 1,647 | 1,925 | 1,890 |

Las diferencias entre variantes son de a lo sumo 0,3 puntos y **cambian de signo entre valid y test** (el MLCA queda levemente arriba en valid en mAP50 y conteo, y levemente abajo en test en mAP50 y mAP50-95).

**Matriz de confusión** (normalizada, valid; casi idéntica en ambas variantes): diagonal 0,97 / 0,96 / 0,93 (*E. coli*, *P. aeruginosa*, *S. aureus*). La confusión entre especies es de a lo sumo 0,01; los errores son sobre todo colonias no detectadas (0,02 / 0,04 / 0,07 en `base_s0`) y falsos positivos. *S. aureus* es la especie más difícil (en `mlca_s0` concentra el 47 % de los falsos positivos).

## 7. Conclusiones

1. **El MLCA no mejora de forma distinguible a YOLOv8m sobre AGAR.** Con 5823 fotos, la diferencia está dentro de lo que se espera por ruido: mismo mAP50, mismo mAP50-95, misma especie por placa y mismo error de conteo, con signo mixto entre valid y test. El resultado es consistente con la versión anterior (102 fotos), donde tampoco se observó mejora.
2. **El modelo más eficiente es YOLOv8m base:** rinde igual con una arquitectura más simple y entrena más rápido (4 h 08 contra 5 h 03 en total; unos 3,3 contra 4,1 min por época, ≈ 25 % más lento el MLCA). La comparación de tiempos supone una GPU similar en todas las sesiones de Colab; no se midió la velocidad de inferencia.
3. **Comparar sobre AGAR fue útil:** con 57 veces más fotos de entrenamiento el resultado se mantiene, lo que sugiere que lo que más pesa son los datos y no el mecanismo de atención.
4. **¿Hacen falta más seeds?** Para elegir el modelo, no: es poco probable que cambien la decisión de usar el base. Para reportar el resultado con respaldo estadístico, convienen las seeds 1 y 2 (`SEEDS = [0, 1, 2]`) y presentar promedio ± desvío sobre test.

## 8. Cómo seguir

- Opcional: ejecutar las seeds 1 y 2 (~9 h por seed, ambas variantes) para informar promedio ± desvío.
- Probar el modelo entrenado en AGAR sobre las fotos propias (dataset v5) para ver si generaliza a otras condiciones de laboratorio.

---
Notebook: `colab_yolov8m_mlca_AGAR.ipynb`. Versión anterior: `RESUMEN_COMPARACION.md`. Presentación: `YOLOv8m_MLCA_AGAR.pptx`.
