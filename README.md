# Identificación y conteo de colonias con YOLOv8

Proyecto **“Desarrollo de sistemas de reconocimiento de imágenes, implementados con inteligencia
artificial, para la integración masiva de datos biológicos”**: Iniciación a la investigación,
Centro Hodos (Universidad Católica del Uruguay – Universidad de la República), 2026.
Dirección: Dr. Armando Borrero Molina y PhD. Germán Traglia.

Detección, conteo y clasificación de colonias de *Escherichia coli* (`col`), *Pseudomonas
aeruginosa* (`osa`) y *Staphylococcus aureus* (`sau`) en fotos de placas de Petri, con YOLOv8m y
con el módulo de atención C2f-MLCA de Colony-YOLO (Wang et al., 2025).

> [Agregar acá una imagen: una placa con las cajas detectadas, p. ej. `results/figuras/ejemplo.jpg`]

## Requisitos

- Python 3.11 o superior y una GPU con CUDA (todo se corrió en una **Tesla T4** de Google Colab).
- Una API key de Roboflow con acceso al workspace `ucuestudiantesproyecto-gmail-com`.
- `ultralytics==8.4.163` exacta: queda fijada en `pyproject.toml`.

## Instalación

```bash
git clone https://github.com/belenkanas/colonias-ucu-udelar.git
cd colonias-ucu-udelar
pip install -e .            # torch: el de Colab, o el de pytorch.org según la GPU
cp .env.example .env        # y pegar la ROBOFLOW_API_KEY
```

En Colab: abrir `notebooks/colab_entrenar_evaluar.ipynb` y cargar la key en los *Secrets*.

## Reproducir los resultados del informe

Cada resultado tiene un archivo de configuración en `configs/experiments/` con el dataset (versión
exacta de Roboflow), el modelo, todos los hiperparámetros y las seeds.

```bash
python -m colonias.train    configs/experiments/v8.yaml
python -m colonias.evaluate configs/experiments/v8.yaml
python scripts/tabla_resultados.py      # compara con results/resultados_esperados.csv
```

| Resultado del informe | Config | Tiempo aprox. (T4) |
|---|---|---|
| Evolución de versiones v1 – v8 (Tabla 2) | `v1.yaml` … `v8.yaml` | 6 min – 2 h c/u |
| YOLOv8m vs. YOLOv8m + C2f-MLCA, dataset v5 (Tabla 3) | `mlca_v5_base.yaml`, `mlca_v5_mlca.yaml` | ~40 min c/u (3 seeds) |
| YOLOv8m vs. YOLOv8m + C2f-MLCA sobre AGAR (Tabla 4) | `agar_base.yaml`, `agar_mlca.yaml` | ~4 h y ~5 h |

Las corridas quedan en `runs/<experimento>_s<seed>/`. Si el entrenamiento se corta, volver a
ejecutar el mismo comando: lo terminado se saltea y lo que quedó a medias se retoma. Con
`--sync <carpeta>` se copia cada corrida a otra carpeta (p. ej. Google Drive) cada 2 épocas.

## Usar los pesos entrenados (sin reentrenar)

Los pesos no están en el repo: se publican en el
[Release `v1.0-informe`](https://github.com/<org>/colonias-ucu-udelar/releases).

| Archivo | Qué es | Config |
|---|---|---|
| `best.pt` | v2 (09/06/2026) | `v2.yaml` |
| `best_v8.pt` | Modelo de referencia v8 (25/09/2026) | `v8.yaml` |
| `mlca_v5_*_s{0,1,2}.pt`, `agar_*_s0.pt` | Corridas de los experimentos MLCA | ver tabla de arriba |

```bash
python -m colonias.evaluate configs/experiments/v8.yaml --pesos best_v8.pt   # métricas del informe
yolo predict model=best_v8.pt source=foto.jpg conf=0.5 iou=0.45              # una foto
```

## Resultados esperados

Valores del informe en `results/resultados_esperados.csv`. Con varias seeds: media ± desvío.

| Experimento | Test mAP50 | Test mAP50-95 | Especie por placa (test) | Error de conteo (test, colonias) |
|---|---|---|---|---|
| YOLOv8m, dataset v5 (3 seeds) | 0,774 ± 0,009 | 0,438 ± 0,018 | 74,2 ± 8,5 % | 1,06 ± 0,18 |
| YOLOv8m + C2f-MLCA, dataset v5 (3 seeds) | 0,769 ± 0,041 | 0,447 ± 0,019 | 65,6 ± 6,7 % | 1,18 ± 0,34 |
| YOLOv8m, AGAR (seed 0) | 0,958 | 0,620 | 96,8 % | 1,93 |
| YOLOv8m + C2f-MLCA, AGAR (seed 0) | 0,955 | 0,617 | 97,0 % | 1,89 |

**Qué significa “reproducir”.** En GPU, dos corridas con la misma seed no dan el mismo resultado
al decimal (el pooling del MLCA, por ejemplo, no tiene implementación determinista en CUDA). Un
reentrenamiento reproduce el resultado si cae dentro del desvío entre seeds. Para obtener las
métricas exactas, evaluar los pesos publicados.

## Datos

Las imágenes no se suben al repo: se descargan de Roboflow en la versión exacta que figura en
`configs/datasets/` (`python -m colonias.data <nombre>`).

| Dataset | Proyecto de Roboflow / versión | Fotos train / valid / test | Usado en |
|---|---|---|---|
| UCU-UDELAR-2026-A | `ucu-udelar-2026-a` / 1 – 3 | 870 / 62 / – (v1) | v1 – v3 |
| Subdataset | `subdataset-ucu-udelar-2026-a` / 1 | 21 / 6 / 6 | v4 |
| Subdataset v2 | `subdataset_v2-ucu-udelar-2026-a` / 2, 4, 5 | 102 / 24 / 31 (v5) | v6 – v8, MLCA |
| AGAR (3 especies) | `agar_subdataset` / 1 | 5823 / 1344 / 1792 | MLCA sobre AGAR |

- AGAR: Majchrowska et al. (2025), licencia CC BY-NC. Convertido a formato Roboflow con
  [AGAR2Roboflow](https://github.com/leapereira/AGAR2Roboflow).
- [Completar: origen de las fotos propias y su licencia.]

## Estructura

```text
configs/datasets/      un .yaml por dataset de Roboflow (proyecto, versión, fotos por split)
configs/experiments/   un .yaml por resultado del informe
src/colonias/          data.py (descarga) · mlca.py (C2f-MLCA) · train.py · evaluate.py
notebooks/             notebook de Colab que llama a src/
scripts/               reproducir_todo.sh · tabla_resultados.py
results/               resultados_esperados.csv · referencia/ (CSV originales)
docs/                  versiones.md (v1 – v8) · experimentos/ (resúmenes de cada experimento)
```

## Trabajos de cada integrante

| Integrante | Trabajo | Dónde |
|---|---|---|
| María Belén Kanas | YOLOv8 v1 – v8, MLCA sobre el dataset v5 y sobre AGAR | este repo |
| Alejandro Piccardo | C2f-MLCA en YOLOv8m; modelo en dos etapas v4.2 | este repo / [completar] |
| Leandro Pereira | Conversor AGAR → Roboflow | [AGAR2Roboflow](https://github.com/leapereira/AGAR2Roboflow) |
| Ignacio Silva | Conteo clásico con transformada de Hough | [trasformadaHough](https://github.com/Tachoviendo/trasformadaHough) |

## Cómo citar

[Completar con la referencia del informe final.]

Referencias principales:
- Wang, M. et al. (2025). Colony-YOLO: A lightweight micro-colony detection network based on
  improved YOLOv8n. *Microorganisms, 13*(7), 1617. https://doi.org/10.3390/microorganisms13071617
- Pintavirooj, C. et al. (2026). Bacterial colony counting and classification system based on deep
  learning model. *Applied Sciences, 16*(3), 1313. https://doi.org/10.3390/app16031313
- Majchrowska, S. et al. (2025). Assessing microbial colony counting: A deep learning approach with
  the AGAR image dataset. *Neurocomputing, 630*, 129654. https://doi.org/10.1016/j.neucom.2025.129654

## Licencia

Código: MIT (`LICENSE`). Los datasets mantienen sus propias licencias.
