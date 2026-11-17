"""C2f-MLCA (Colony-YOLO, Wang et al., 2025) integrado en YOLOv8m.

Código tomado sin cambios de lógica de los notebooks colab_yolov8m_mlca.ipynb y
colab_yolov8m_mlca_AGAR.ipynb.

* MLCA (Mixed Local Channel Attention): atención de canales local (pooling 5x5) + global,
  cada una con un Conv1d de kernel k según la Ec. (3) del paper (gamma = b = 2).
* Bottleneck_MLCA: x -> Conv -> Conv -> MLCA (-> + x si hay shortcut). Figura 6 del paper.
* C2f_MLCA: un C2f cuyos Bottleneck son Bottleneck_MLCA.

ultralytics solo calcula canales y repeticiones para los módulos de una lista interna.
Mientras se construye el modelo, el nombre C2fPSA (que está en esa lista y YOLOv8m no usa)
apunta a C2f_MLCA. Así solo llevan atención las capas que el YAML declara como C2f_MLCA
y el backbone queda intacto. Probado con ultralytics==8.4.163.
"""

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
import ultralytics
import ultralytics.nn.modules.block as block
import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules.block import C2f, Bottleneck


class MLCA(nn.Module):
    def __init__(self, channel, local_size=5, gamma=2, b=2, local_weight=0.5):
        super().__init__()
        self.local_size = int(local_size)
        self.local_weight = float(local_weight)
        c = int(channel)
        if c % 2 == 0:  # Ec. (3): si C es par, C+1
            c += 1
        t = int(abs(math.log2(c) / gamma + b / gamma))
        k = max(t if t % 2 else t + 1, 3)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.local_pool = nn.AdaptiveAvgPool2d(self.local_size)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.shape
        ls = self.local_size
        local = self.local_pool(x)
        glob = self.global_pool(local)
        t_local = local.view(b, c, -1).transpose(-1, -2).reshape(b, 1, -1)
        t_glob = glob.view(b, c, -1).transpose(-1, -2)
        y_local = self.conv_local(t_local)
        y_glob = self.conv(t_glob)
        att_local = y_local.reshape(b, ls * ls, c).transpose(-1, -2).reshape(b, c, ls, ls).sigmoid()
        att_glob = y_glob.transpose(-1, -2).reshape(b, c, 1, 1).sigmoid().expand(-1, -1, ls, ls)
        att = att_glob * (1 - self.local_weight) + att_local * self.local_weight
        return x * F.interpolate(att, size=(h, w), mode="nearest")


class Bottleneck_MLCA(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.mlca = MLCA(c2)

    def forward(self, x):
        y = self.mlca(self.cv2(self.cv1(x)))
        return x + y if self.add else y


class C2f_MLCA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            Bottleneck_MLCA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )


def registrar() -> None:
    """Registra los módulos en ultralytics. Hay que llamarla antes de crear o cargar un modelo MLCA."""
    # mismo nombre de módulo en Colab y en local: los .pt se pueden cargar en los dos lados
    for cls in (MLCA, Bottleneck_MLCA, C2f_MLCA):
        cls.__module__ = "ultralytics.nn.modules.block"
        cls.__qualname__ = cls.__name__
        setattr(block, cls.__name__, cls)
        setattr(tasks, cls.__name__, cls)

    if getattr(tasks.parse_model, "_mlca_v8", False):
        return
    assert "C2fPSA" in inspect.getsource(tasks.parse_model), (
        "Esta versión de ultralytics cambió parse_model: usar ultralytics==8.4.163")
    original = tasks.parse_model

    def parse_model(d, ch, verbose=True):
        saved = tasks.C2fPSA
        tasks.C2fPSA = C2f_MLCA
        try:
            return original(d, ch, verbose)
        finally:
            tasks.C2fPSA = saved

    parse_model._mlca_v8 = True
    tasks.parse_model = parse_model


def yaml_mlca(destino: str | Path, nc: int = 3) -> str:
    """Escribe yolov8m-mlca.yaml: el yolov8.yaml de ultralytics con los 4 C2f del neck -> C2f_MLCA.

    El nombre del archivo tiene que empezar con 'yolov8m' para que ultralytics use la escala m.
    """
    src = Path(ultralytics.__file__).parent / "cfg" / "models" / "v8" / "yolov8.yaml"
    out, in_head, n = [], False, 0
    for ln in src.read_text().splitlines():
        if ln.startswith("head:"):
            in_head = True
        if in_head and ", C2f, [" in ln:
            ln, n = ln.replace(", C2f, [", ", C2f_MLCA, ["), n + 1
        if ln.startswith("nc:"):
            ln = f"nc: {nc}"
        out.append(ln)
    assert n == 4, f"se esperaban 4 C2f en el neck y se encontraron {n}"
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(out) + "\n")
    return str(destino)


def construir_modelo(variante: str, pesos: str = "yolov8m.pt", nc: int = 3, carpeta: str = "runs"):
    """'base' = YOLOv8m tal cual. 'mlca' = YOLOv8m con C2f-MLCA en el neck y pesos COCO en el resto."""
    from ultralytics import YOLO

    if variante == "base":
        return YOLO(pesos)
    if variante != "mlca":
        raise ValueError(f"variante desconocida: {variante}")
    registrar()
    m = YOLO(yaml_mlca(os.path.join(carpeta, "yolov8m-mlca.yaml"), nc=nc))
    m.load(pesos)  # COCO en todo menos el MLCA (469 de 491 tensores)
    return m


def verificar(modelo, variante: str) -> dict:
    """Cuenta módulos: base -> C2f=8, MLCA=0. mlca -> C2f=4, C2f_MLCA=4, MLCA=8, 0 en el backbone."""
    from collections import Counter

    k = Counter(type(x).__name__ for x in modelo.model.modules())
    bb = Counter(type(x).__name__ for x in modelo.model.model[:10].modules())
    info = {"C2f": k["C2f"], "C2f_MLCA": k["C2f_MLCA"], "MLCA": k["MLCA"], "MLCA_backbone": bb["MLCA"]}
    if variante == "mlca":
        assert info["C2f_MLCA"] == 4 and info["MLCA"] == 8 and info["MLCA_backbone"] == 0, info
    return info
