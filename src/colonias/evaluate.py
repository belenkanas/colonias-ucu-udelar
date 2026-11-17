"""Evalúa las corridas de un experimento (o pesos sueltos) con las métricas del informe.

    python -m colonias.evaluate configs/experiments/v8_mlca.yaml
    python -m colonias.evaluate configs/experiments/v8.yaml --pesos best_v8.pt   # pesos del Release

Métricas, sobre valid y test:
* mAP50 y mAP50-95 con model.val (conf 0,001; IoU 0,7: lo estándar de ultralytics).
* Especie por placa: voto mayoritario de las cajas detectadas (conf 0,5; NMS 0,45).
  La especie verdadera sale del nombre del archivo (dataset propio: "ecoli", "saureus",
  "paeruginosa") o de la clase mayoritaria de las etiquetas (AGAR, nombres numéricos).
* conteo_MAE: error absoluto medio entre colonias detectadas y cajas anotadas por foto.
  Ojo: se compara contra cajas anotadas, no contra UFC reales.

Salida: results/<nombre>.csv, una fila por corrida (mismo formato que resultados_colab.csv).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from colonias.data import RAIZ, descargar

EXT = (".jpg", ".jpeg", ".png")


def especie_por_nombre(nombre: str, names: list[str]) -> int | None:
    n = nombre.lower()
    clave = ("col" if "ecoli" in n else "sau" if "saureus" in n
             else "osa" if ("paureginosa" in n or "paeruginosa" in n) else None)
    return names.index(clave) if clave in names else None


def leer_etiquetas(lab: Path) -> list[int]:
    if not lab.exists():
        return []
    return [int(l.split()[0]) for l in lab.read_text().splitlines() if len(l.split()) == 5]


def placas(model, carpeta: Path, names: list[str], especie_desde: str, conf: float, iou: float, imgsz: int):
    fotos = sorted(p for p in (carpeta / "images").iterdir() if p.suffix.lower() in EXT)
    ok, err = [], []
    for i in range(0, len(fotos), 32):
        lote = fotos[i:i + 32]
        for f, r in zip(lote, model.predict([str(x) for x in lote], imgsz=imgsz, conf=conf, iou=iou, verbose=False)):
            gt = leer_etiquetas(carpeta / "labels" / f"{f.stem}.txt")
            if especie_desde == "etiquetas":
                esp = int(np.bincount(gt, minlength=len(names)).argmax()) if gt else None
            else:
                esp = especie_por_nombre(f.name, names)
            cl = r.boxes.cls.cpu().numpy().astype(int)
            voto = int(np.bincount(cl, minlength=len(names)).argmax()) if len(cl) else None
            ok.append(voto is not None and voto == esp)
            err.append(len(cl) - len(gt))
    return float(np.mean(ok)), float(np.mean(np.abs(err)))


def evaluar(cfg: dict, corridas: list[tuple[str, Path]]) -> pd.DataFrame:
    from ultralytics import YOLO

    if cfg.get("variante") == "mlca":
        from colonias.mlca import registrar
        registrar()

    data_yaml = descargar(cfg["dataset"])
    d = yaml.safe_load(Path(data_yaml).read_text())
    raiz, names = Path(d["path"]), list(d["names"])
    ev = cfg.get("evaluacion", {})
    imgsz = cfg["entrenamiento"].get("imgsz", 640)

    filas = []
    for nombre, pesos in corridas:
        m = YOLO(str(pesos))
        fila = {"experimento": cfg["nombre"], "corrida": nombre, "variante": cfg.get("variante", "base")}
        for split, carpeta in (("val", "valid"), ("test", "test")):
            if not (raiz / carpeta / "images").exists():
                continue
            r = m.val(data=data_yaml, split=split, imgsz=imgsz, batch=16, conf=0.001, iou=0.7,
                      plots=False, verbose=False)
            fila[f"{split}_mAP50"] = float(r.box.map50)
            fila[f"{split}_mAP50_95"] = float(r.box.map)
            fila[f"{split}_especie"], fila[f"{split}_conteo_MAE"] = placas(
                m, raiz / carpeta, names, ev.get("especie_desde", "nombre_archivo"),
                ev.get("conf", 0.5), ev.get("iou_nms", 0.45), imgsz)
        filas.append(fila)
        print(nombre, {k: round(v, 3) for k, v in fila.items() if isinstance(v, float)})
    return pd.DataFrame(filas)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--project", default="runs")
    p.add_argument("--pesos", nargs="+", help="evaluar estos .pt en lugar de las corridas en runs/")
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())

    if a.pesos:
        corridas = [(Path(x).stem, Path(x)) for x in a.pesos]
    else:
        corridas = [(d.name, d / "weights" / "best.pt")
                    for d in sorted(Path(a.project).glob(f"{cfg['nombre']}_s*"))
                    if (d / "weights" / "best.pt").exists()]
    if not corridas:
        raise SystemExit("No hay pesos para evaluar: entrenar primero o pasar --pesos.")

    df = evaluar(cfg, corridas)
    salida = RAIZ / "results" / f"{cfg['nombre']}.csv"
    salida.parent.mkdir(exist_ok=True)
    df.to_csv(salida, index=False)
    print("guardado en", salida)


if __name__ == "__main__":
    main()
