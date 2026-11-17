# Notebooks originales

Copias de los notebooks de Colab tal como se usaron, **sin salidas y sin la API key**, para
trazabilidad. Para reproducir resultados usar `notebooks/colab_entrenar_evaluar.ipynb` o los
scripts de `src/`, que corren el mismo entrenamiento a partir de `configs/experiments/`.

| Notebook | Versión / experimento | Config equivalente |
|---|---|---|
| `train_yolo_colab_v1.ipynb` | v1 (plantilla del curso; v2 – v6 cambian solo épocas, tamaño y dataset) | `v1.yaml` … `v6.yaml` |
| `train_yolo_colab_v7.ipynb` | v7 | `v7.yaml` |
| `train_yolo_colab_v8.ipynb` | v8 (`best_v8.pt`) | `v8.yaml` |
| `colab_yolov8m_mlca.ipynb` | MLCA sobre el dataset v5, 3 seeds | `mlca_v5_base.yaml`, `mlca_v5_mlca.yaml` |
| `colab_yolov8m_mlca_AGAR.ipynb` | MLCA sobre AGAR | `agar_base.yaml`, `agar_mlca.yaml` |
