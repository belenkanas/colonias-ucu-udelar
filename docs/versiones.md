# Historial de versiones (Belén Kanas)

Qué cambió en cada versión, por qué, y con qué config se reproduce. Las conclusiones vienen de
las presentaciones `Kanas - Presentacion v1` a `v6`, de los notebooks `train_yolo_colab_v7/v8` y
de los `train_args` guardados dentro de `best.pt` y `best_v8.pt`.

| Versión | Fecha | Config | Dataset (Roboflow) | Cambio principal | Valid mAP50 / mAP50-95 |
|---|---|---|---|---|---|
| v1 | 08/06/2026 | `v1.yaml` | ucu-udelar-2026-a / 1 | Línea base YOLOv8m, 50 épocas, 640 px | 0,404 / 0,224 |
| v2 | 09/06/2026 | `v2.yaml` | ucu-udelar-2026-a / 2 | 100 épocas, rebalance 70/15/15, stretch 640 | 0,478 / 0,252 |
| v3 | 11/06/2026 | `v3.yaml` | ucu-udelar-2026-a / 3 | YOLOv8l, 80/10/10, rotación ±45°, auto-contraste | completar |
| v4 | 17/06/2026 | `v4.yaml` | subdataset-ucu-udelar-2026-a / 1 | Subdataset de 33 imágenes (10 por clase) | completar |
| v5 | 18/06/2026 | `v5.yaml` | confirmar | Todas las imágenes con < 30 colonias (165) | completar |
| v6 | 02/07/2026 | `v6.yaml` | subdataset_v2-ucu-udelar-2026-a / 2 | ~30 colonias por placa, clases balanceadas | completar |
| v7 | ≈ 25/09/2026 | `v7.yaml` | subdataset_v2-ucu-udelar-2026-a / 4 | Receta de Pintavirooj et al.: Adam, lr 1e-4, 10 épocas, 224 px | 0,106 / 0,040 |
| v8 | 25/09/2026 | `v8.yaml` | subdataset_v2-ucu-udelar-2026-a / 5 | Receta corregida: 100 épocas, 640 px, augmentaciones del paper | 0,928 / 0,608 |

## Notas por versión

- **v1.** Se intentó 100 épocas a 896 px, pero se quedaba sin memoria (CUDA OutOfMemory). El dataset
  tenía anotaciones mixtas (cajas + segmentación) y se observaron cajas superpuestas.
- **v2.** Mejoran col (+8 %) y osa (+55 %), pero sau cae de 0,747 a 0,555 de mAP50. 281 col se
  clasifican como sau. Configuración verificada en `best.pt`: batch 16, optimizer auto, patience 100.
- **v3.** El modelo large no mejoró: aumentaron las confusiones col/sau. El rebalance 80/10/10 empeoró.
- **v4 – v5.** Con placas contables los errores se interpretan mejor, pero aparece una sobrecarga de sau
  (1343 cajas en v5). Las presentaciones dicen "Large (m)": el modelo es medium.
- **v6.** Desaparece la sobrecarga de clases en la matriz de confusión. Casos puntuales:
  IMG_8787 (col) 13/13 e IMG_7819 (osa) 27/32 con la especie correcta.
- **v7.** Primera adaptación del paper. No convergió en 10 épocas; además, 224 px es el tamaño del paper
  para el recorte de una sola colonia, no para la placa completa, y `conf=0.4` / `iou=0.5` en el
  entrenamiento también filtraban la validación de cada época.
- **v8.** Se corrigió todo lo anterior (comentarios en `train_yolo_colab_v8.ipynb`): 100 épocas,
  640 px, patience 15, escala 0,9 – 1,1, brillo ±15 %, sin cambios de matiz ni saturación (el color
  es un rasgo de la especie). Valid: P 0,916, R 0,867. **Ojo:** valid también eligió `best.pt`.

## Después de v8

- **MLCA sobre el dataset v5** (`mlca_v5_base.yaml`, `mlca_v5_mlca.yaml`): 3 seeds en Colab.
  Resultados en `results/referencia/mlca_v5_colab.csv`. Es la tabla de la presentación del 16/10/2026.
  El `RESUMEN_COMPARACION.md` suma además 3 corridas locales con batch 16 (6 por variante).
- **MLCA sobre AGAR** (`agar_base.yaml`, `agar_mlca.yaml`): ver `docs/experimentos/RESUMEN_AGAR.md`.
