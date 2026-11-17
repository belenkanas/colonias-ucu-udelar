"""Descarga de datasets desde Roboflow según configs/datasets/<nombre>.yaml.

La API key nunca se escribe en el código. Se busca, en este orden:
1. la variable de entorno ROBOFLOW_API_KEY,
2. el archivo .env en la raíz del repo (línea ROBOFLOW_API_KEY=...),
3. los Secrets de Colab (ícono de llave, nombre ROBOFLOW_API_KEY),
4. y si no está, se pide por teclado.

Uso:  python -m colonias.data subdataset_v2_v5
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]
CONFIGS = RAIZ / "configs" / "datasets"
DATA_DIR = Path(os.environ.get("COLONIAS_DATA", RAIZ / "data"))


def _api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    env = RAIZ / ".env"
    if not key and env.exists():
        for ln in env.read_text().splitlines():
            if ln.strip().startswith("ROBOFLOW_API_KEY="):
                key = ln.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        try:
            from google.colab import userdata  # type: ignore
            key = userdata.get("ROBOFLOW_API_KEY")
        except Exception:
            key = None
    if not key:
        key = getpass.getpass("API key de Roboflow: ")
    return key.strip()


def config_dataset(nombre: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{nombre}.yaml").read_text())


def descargar(nombre: str) -> str:
    """Baja la versión exacta del dataset (si no está) y devuelve la ruta a su data yaml."""
    cfg = config_dataset(nombre)
    destino = DATA_DIR / nombre
    if not (destino / "data.yaml").exists():
        from roboflow import Roboflow

        rf = Roboflow(api_key=_api_key())
        (rf.workspace(cfg["workspace"]).project(cfg["proyecto"])
           .version(int(cfg["version"])).download("yolov8", location=str(destino)))

    # las clases se leen del data.yaml descargado, para que coincidan con los ids de las etiquetas
    names = yaml.safe_load((destino / "data.yaml").read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    data_yaml = DATA_DIR / f"{nombre}.yaml"
    data_yaml.write_text(yaml.safe_dump(
        {"path": str(destino), "train": "train/images", "val": "valid/images",
         "test": "test/images", "nc": len(names), "names": names}, sort_keys=False))

    # control: las cantidades tienen que coincidir con las registradas en el config
    esperado = cfg.get("fotos", {})
    for split in ("train", "valid", "test"):
        carpeta = destino / split / "images"
        n = len(list(carpeta.iterdir())) if carpeta.exists() else 0
        if split in esperado and n != esperado[split]:
            print(f"AVISO {nombre}/{split}: {n} fotos, el config dice {esperado[split]}")
        else:
            print(f"{nombre}/{split}: {n} fotos")
    return str(data_yaml)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        print(descargar(n))
