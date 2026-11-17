"""Junta los CSV de results/ y compara con results/resultados_esperados.csv.

    python scripts/tabla_resultados.py

Para cada experimento y métrica muestra la media ± desvío de las corridas propias y el valor
esperado. "OK" = la diferencia está dentro de 2 desvíos entre seeds (o dentro de 0,02 si
hay una sola corrida): con GPU no se obtienen resultados idénticos al decimal, ni con la
misma seed.
"""

from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
METRICAS = ["val_mAP50", "val_mAP50_95", "val_especie", "val_conteo_MAE",
            "test_mAP50", "test_mAP50_95", "test_especie", "test_conteo_MAE"]

esperados = pd.read_csv(RES / "resultados_esperados.csv")
propios = [pd.read_csv(f) for f in sorted(RES.glob("*.csv")) if f.name != "resultados_esperados.csv"]
if not propios:
    raise SystemExit("No hay resultados en results/: correr primero python -m colonias.evaluate <config>")
df = pd.concat(propios, ignore_index=True)

filas = []
for exp, g in df.groupby("experimento"):
    for m in METRICAS:
        if m not in g or g[m].isna().all():
            continue
        e = esperados[(esperados.experimento == exp) & (esperados.metrica == m)]
        media, desv = g[m].mean(), g[m].std() if len(g) > 1 else float("nan")
        ref = e.media.iloc[0] if len(e) else float("nan")
        tol = 2 * (e.desvio.iloc[0] if len(e) and pd.notna(e.desvio.iloc[0]) else 0.01)
        filas.append({"experimento": exp, "metrica": m, "n": len(g), "obtenido": round(media, 3),
                      "desvio": round(desv, 3), "esperado": ref,
                      "estado": "OK" if abs(media - ref) <= tol else ("sin referencia" if pd.isna(ref) else "REVISAR")})

out = pd.DataFrame(filas)
print(out.to_string(index=False))
out.to_csv(RES / "comparacion_con_esperados.csv", index=False)
