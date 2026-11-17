# v4.2 vs YOLOv8m de Belén, y la prueba con MLCA

## 1. Qué diferencia a los dos modelos

| | **v4.2 (nuestro)** | **YOLOv8m (Belén)** |
|---|---|---|
| enfoque | dos etapas: un detector cuenta y un clasificador decide la especie de la placa | un solo detector que cuenta y clasifica (3 clases) |
| red | Colony-YOLO (StarNet-S4 + C2f-MLCA + Shape-IoU), 10,1 M + ResNet-18 | YOLOv8m pre-entrenado en COCO, 25,9 M |
| entrada | placa recortada a 2250 px, detector a 1280 | foto completa estirada a 640×640 |
| datos | UCU_UdelaR v13: 323 fotos de train, ~34 colonias por foto (máx. 595) | subdataset v5: 102 fotos de train, ~10 colonias por foto (máx. 45) |
| especie | promedio de las colonias de la placa | voto de las cajas |

- Los datasets comparten 65 fotos. En ellas, Belén marcó 598 cajas y nosotros 525, y **370 coinciden**: ella marca ~14 % más.
- 92 de sus 157 fotos no están en el nuestro.

## 2. Comparación, en los dos sentidos

Cada modelo se probó con las fotos y anotaciones del otro, con su propia configuración y las mismas métricas. En "sus fotos" no entran las que v4.2 usó para entrenar.

![v4.2 vs YOLOv8m](evaluacion/grafico.png)

| | nuestras 69 fotos (nuestras anotaciones) | | sus 39 fotos (sus anotaciones) | |
|---|---|---|---|---|
| | **v4.2** | YOLOv8m | v4.2 (a 640 px) | **YOLOv8m** |
| especie por placa | **90 %** | 39 % | 26 % | **90 %** |
| error de conteo (colonias) | **3,5** | 11,5 | **1,3** | 1,7 |
| AP50 sin especie | **0,47** | 0,25 | 0,76 | **0,92** |

- **Conteo: gana v4.2.** En nuestras fotos la diferencia es grande y significativa: 8 colonias menos de error en promedio, IC95 [4,3; 13,0]. En las suyas empata, aun recibiendo la foto a 640 px (IC95 [−1,6; +0,6]).
- **Especie: cada modelo solo acierta en su propio dataset.**
  - Su modelo, en nuestras fotos, acierta el 54 % si la condición de laboratorio estaba en su train y el 30 % si no estaba.
  - v4.2 falla en sus fotos por la resolución. Con nuestras fotos bajadas a 640 px cae del 90 % al 49 %, y en las 7 fotos suyas que tenemos en alta resolución acierta 7 de 7.
- **Su mAP50 de 0,93 viene del dataset, no de la arquitectura.** Son placas ralas, y en sus fotos v4.2 también sube a 0,76 de AP50.

## 3. Agregar el MLCA (atención del paper) a su modelo

Probé su YOLOv8m con los 4 C2f del neck cambiados por C2f-MLCA. Todo lo demás quedó igual: sus datos y sus hiperparámetros. Hice 6 corridas por variante: 3 acá con batch 16 y 3 en Colab con batch 32, como ella.

![YOLOv8m vs YOLOv8m + MLCA](yolov8m_mlca/grafico.png)

| en su test (31 fotos) | YOLOv8m | YOLOv8m + MLCA | diferencia |
|---|---|---|---|
| mAP50 (3 clases) | 0,737 ± 0,025 | 0,752 ± 0,039 | +0,015 ± 0,019 (ruido) |
| especie por placa | 65 % | 65 % | 0 |
| error de conteo | 1,57 | 1,80 | **cambia de signo**: −0,32 acá, +0,77 en Colab |

- **El MLCA no mejora.** El mAP sube un poco, pero dentro del ruido entre seeds. La especie no cambia.
- **El conteo empeora con la configuración exacta de ella** (+0,77 colonias en Colab, 4,7 veces su error estándar).
- **En nuestras fotos tampoco ayuda:** la especie queda en 37 % con MLCA y 42 % sin MLCA.
- **Su corrida original fue favorable.** Tuvo mAP50 0,778 y 81 % de especie, por encima de los 6 reentrenamientos: 0,696 – 0,761 y 61 – 71 %, incluido uno con su misma seed. **Lo esperable de su modelo es ~0,74 y ~65 %.**

## 4. Cómo se comparan

- **Para contar:** v4.2 > YOLOv8m ≈ YOLOv8m + MLCA.
- **Para la especie:** ninguno está probado en condiciones de laboratorio nuevas. Todos los 65 – 90 % salen de fotos de las mismas sesiones que el train.
- **El MLCA no aporta en su modelo.** Lo que más pesa son los datos (densidad de las placas, regla de anotación, resolución), no la atención.

**Cómo seguir:**
1. Unir los dos datasets en resolución original (sus 92 fotos nuevas + las 461 nuestras).
2. Acordar una sola regla de anotación.
3. Armar un test común separado por condición de laboratorio.
4. Reportar siempre el promedio de varias seeds sobre el test, no una corrida sobre validación.

---
Detalle: [`evaluacion/COMPARACION.md`](evaluacion/COMPARACION.md) y [`yolov8m_mlca/RESULTADOS.md`](yolov8m_mlca/RESULTADOS.md).
