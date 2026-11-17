"""Entrena un experimento a partir de su config.

    python -m colonias.train configs/experiments/v8.yaml
    python -m colonias.train configs/experiments/agar_mlca.yaml --seeds 0 1 2 --sync /content/drive/MyDrive/colonias

* Una corrida por seed: runs/<nombre>_s<seed>/ (pesos, results.csv, gráficas).
* Las corridas con marca TERMINADO se saltean; si hay un last.pt a medias, se retoma (resume=True).
* --sync copia la corrida a otra carpeta (p. ej. Google Drive) cada --sync-every épocas.
  En Colab, entrenar directo en Drive falla (OSError Errno 95): se entrena en disco local y se copia.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import yaml

from colonias.data import descargar


def leer_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def sync(src: Path, dst: Path) -> None:
    """Copia src -> dst (solo lo que cambió). Cada archivo va a un .tmp y se renombra."""
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out = dst if rel == "." else dst / rel
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            s, d = Path(root) / f, out / f
            if d.exists() and d.stat().st_size == s.stat().st_size and d.stat().st_mtime >= s.stat().st_mtime:
                continue
            shutil.copy2(s, str(d) + ".tmp")
            os.replace(str(d) + ".tmp", d)


def _callback(local: Path, dst: Path, cada: int):
    def cb(trainer):  # se llama justo después de guardar last.pt / best.pt
        if (trainer.epoch + 1) % cada == 0:
            try:
                sync(local, dst)
                print(f"   [sync] época {trainer.epoch + 1} copiada a {dst}")
            except Exception as e:  # una falla de copia no corta el entrenamiento
                print("   [sync] aviso:", e)
    return cb


def _tiene_optimizador(ckpt: Path) -> bool:
    import torch

    return torch.load(ckpt, map_location="cpu", weights_only=False).get("optimizer") is not None


def entrenar(cfg: dict, seeds: list[int], project: str, sync_dir: str | None, sync_every: int) -> None:
    from ultralytics import YOLO

    from colonias.mlca import construir_modelo, registrar, verificar

    data_yaml = descargar(cfg["dataset"])
    variante = cfg.get("variante", "base")
    if variante == "mlca":
        registrar()

    for seed in seeds:
        nombre = f"{cfg['nombre']}_s{seed}"
        local = Path(project).resolve() / nombre
        dst = Path(sync_dir) / nombre if sync_dir else None
        if dst and not (local / "weights" / "last.pt").exists() and (dst / "weights" / "last.pt").exists():
            sync(dst, local)  # se perdió el disco local (reinicio de Colab): restaurar desde Drive
        if (local / "TERMINADO").exists():
            print("ya estaba:", nombre)
            continue

        t0 = time.time()
        last = local / "weights" / "last.pt"
        if last.exists() and not _tiene_optimizador(last):
            print(">>>", nombre, "ya había terminado de entrenar")  # ultralytics borra el optimizador al final
        elif last.exists():
            print(">>> retomando", nombre)
            m = YOLO(str(last))
            if dst:
                m.add_callback("on_model_save", _callback(local, dst, sync_every))
            m.train(resume=True)
        else:
            print(">>> entrenando", nombre)
            m = construir_modelo(variante, cfg.get("modelo", "yolov8m.pt"), carpeta=project)
            if variante == "mlca":
                print("   módulos:", verificar(m, variante))
            if dst:
                m.add_callback("on_model_save", _callback(local, dst, sync_every))
            m.train(data=data_yaml, seed=seed, project=str(local.parent), name=nombre,
                    exist_ok=True, plots=True, **cfg["entrenamiento"])

        (local / "TERMINADO").write_text("ok")
        if dst:
            sync(local, dst)
        print(f">>> {nombre} listo en {(time.time() - t0) / 60:.1f} min")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--seeds", type=int, nargs="+", help="por defecto, las del config")
    p.add_argument("--project", default="runs")
    p.add_argument("--sync", dest="sync_dir", help="carpeta donde copiar las corridas (p. ej. Drive)")
    p.add_argument("--sync-every", type=int, default=2)
    a = p.parse_args()
    cfg = leer_config(a.config)
    entrenar(cfg, a.seeds or cfg.get("seeds", [0]), a.project, a.sync_dir, a.sync_every)


if __name__ == "__main__":
    main()
