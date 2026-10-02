"""Linea base: predecir el techo declarado con estadisticos simples de la imagen.

Por que existe. Antes de ajustar una red neuronal hay que saber contra que se
compara. Esta linea base no mira formas ni texturas aprendidas: resume cada foto
en color, brillo, contraste y densidad de bordes, y le pasa esos numeros a una
regresion logistica. Es barata, corre en CPU en minutos y da el piso que
cualquier modelo de vision tiene que superar para justificar su costo.

Tiene un segundo uso, mas importante que el primero. Si esta linea base ya
acierta mucho, la sospecha no es que el problema sea facil sino que hay fuga: un
techo de lamina brillante y una foto sobreexpuesta son cosas distintas, pero en
un histograma se parecen, y el brillo puede estar correlacionado con la region y
la region con el estado declarado. Por eso todo se evalua agrupando por
municipio, la regla que fijo school_infrastructure.

Como se agrega. La etiqueta es de la sede, no de la foto. El modelo puntua cada
foto y despues se promedia por sede, que es la unidad que le importa a quien
decide. Se reportan las dos cosas.

Lo que NO hace. No afirma nada sobre el estado real del edificio y no reemplaza
la anotacion ordinal pendiente.

Uso:
    uv run python scripts/03_linea_base.py
    uv run python scripts/03_linea_base.py --experimento techo_malo

Salidas:
    mlruns/                                   corridas registradas en MLflow
    results/perfilado/03_linea_base.txt       el resumen en texto
    data/derived/rasgos_basicos.parquet       los rasgos, para no recalcularlos
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "data" / "derived" / "fotos.parquet"
MANIFIESTO = RAIZ / "data" / "derived" / "fotos_descargadas.csv"
RASGOS = RAIZ / "data" / "derived" / "rasgos_basicos.parquet"

SEMILLA = 20260923
PLIEGUES = 5


def rasgos_de(ruta: Path) -> dict | None:
    """Resume una foto en numeros: color, brillo, contraste y bordes."""
    try:
        img = Image.open(ruta).convert("RGB")
    except Exception:                                   # noqa: BLE001
        return None

    arr = np.asarray(img, dtype=np.float32) / 255.0
    gris = arr.mean(axis=2)

    d = {
        "alto_px": arr.shape[0],
        "ancho_px": arr.shape[1],
        "relacion_aspecto": arr.shape[1] / arr.shape[0],
        "brillo": float(gris.mean()),
        "contraste": float(gris.std()),
        "oscuros": float((gris < 0.2).mean()),
        "quemados": float((gris > 0.9).mean()),
    }
    for i, canal in enumerate(["r", "g", "b"]):
        d[f"{canal}_media"] = float(arr[:, :, i].mean())
        d[f"{canal}_std"] = float(arr[:, :, i].std())

    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    d["saturacion"] = float(hsv[:, :, 1].mean())
    d["saturacion_std"] = float(hsv[:, :, 1].std())
    # Histograma de tono: separa el verde de la vegetacion del gris del concreto
    hist, _ = np.histogram(hsv[:, :, 0], bins=8, range=(0, 1), density=True)
    for i, v in enumerate(hist):
        d[f"tono_{i}"] = float(v)

    bordes = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    d["bordes_media"] = float(bordes.mean())
    d["bordes_std"] = float(bordes.std())
    d["bordes_fuertes"] = float((bordes > 0.25).mean())
    return d


def calcula_rasgos(fotos: pd.DataFrame) -> pd.DataFrame:
    """Recorre las fotos en disco y arma la tabla de rasgos, con cache."""
    if RASGOS.exists():
        previo = pd.read_parquet(RASGOS)
        faltan = fotos[~fotos.set_index(["cod_dane", "n_foto"]).index.isin(
            previo.set_index(["cod_dane", "n_foto"]).index)]
    else:
        previo, faltan = pd.DataFrame(), fotos

    nuevos = []
    for i, (_, f) in enumerate(faltan.iterrows(), 1):
        d = rasgos_de(RAIZ / f.archivo)
        if d is not None:
            nuevos.append({"cod_dane": f.cod_dane, "n_foto": f.n_foto, **d})
        if i % 250 == 0:
            print(f"  rasgos {i:,}/{len(faltan):,}", flush=True)

    tabla = pd.concat([previo, pd.DataFrame(nuevos)], ignore_index=True) if nuevos else previo
    tabla.to_parquet(RASGOS, index=False)
    return tabla


def evalua(y, p, grupo_sede) -> dict:
    """Metricas por foto y por sede. La sede es la unidad de decision."""
    m = {
        "auc_foto": roc_auc_score(y, p),
        "pr_auc_foto": average_precision_score(y, p),
        "f1_foto": f1_score(y, (p > 0.5).astype(int)),
    }
    por_sede = pd.DataFrame({"sede": grupo_sede, "y": y, "p": p}).groupby("sede").mean()
    m["auc_sede"] = roc_auc_score(por_sede.y, por_sede.p)
    m["pr_auc_sede"] = average_precision_score(por_sede.y, por_sede.p)
    m["base_positivos"] = float(np.mean(y))
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experimento", default="techo_malo")
    args = ap.parse_args()

    if not MANIFIESTO.exists():
        raise SystemExit("Falta el manifiesto de descarga. Corre antes el script 02.")

    base = pd.read_parquet(BASE)
    man = pd.read_csv(MANIFIESTO)
    man = man[man.estado.isin(["ok", "ya_estaba"])]

    fotos = base.merge(man[["cod_dane", "n_foto", "archivo"]], on=["cod_dane", "n_foto"])
    fotos = fotos[fotos.particion == "entrenamiento"].copy()
    print(f"fotos en disco para entrenar: {len(fotos):,} de {fotos.cod_dane.nunique():,} sedes")

    rasgos = calcula_rasgos(fotos)
    datos = fotos.merge(rasgos, on=["cod_dane", "n_foto"])
    columnas = [c for c in rasgos.columns if c not in ("cod_dane", "n_foto")]

    X = datos[columnas].to_numpy()
    y = datos["techo_malo"].to_numpy()
    grupos = datos["mpio_id"].to_numpy()

    mlflow.set_tracking_uri(f"file:///{(RAIZ / 'mlruns').as_posix()}")
    mlflow.set_experiment(args.experimento)

    with mlflow.start_run(run_name="linea_base_estadisticos"):
        mlflow.log_params({
            "modelo": "logistica sobre estadisticos de imagen",
            "n_rasgos": len(columnas),
            "n_fotos": len(datos),
            "n_sedes": int(datos.cod_dane.nunique()),
            "n_municipios": int(datos.mpio_id.nunique()),
            "validacion": f"GroupKFold por municipio, {PLIEGUES} pliegues",
            "semilla": SEMILLA,
        })

        pred = np.zeros(len(y))
        for entrena, prueba in GroupKFold(n_splits=PLIEGUES).split(X, y, groups=grupos):
            modelo = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            )
            modelo.fit(X[entrena], y[entrena])
            pred[prueba] = modelo.predict_proba(X[prueba])[:, 1]

        metricas = evalua(y, pred, datos["cod_dane"].to_numpy())
        mlflow.log_metrics(metricas)

        salida = datos[["cod_dane", "n_foto", "mpio_id", "techo_malo"]].copy()
        salida["p"] = pred
        tmp = RAIZ / "data" / "derived" / "pred_linea_base.csv"
        salida.to_csv(tmp, index=False)
        mlflow.log_artifact(str(tmp))

    lineas = [
        "linea base: regresion logistica sobre estadisticos de imagen",
        f"fotos {len(datos):,}   sedes {datos.cod_dane.nunique():,}   "
        f"municipios {datos.mpio_id.nunique():,}",
        f"validacion: GroupKFold por municipio, {PLIEGUES} pliegues",
        "",
    ]
    lineas += [f"  {k:<16} {v:.3f}" for k, v in metricas.items()]
    lineas += [
        "",
        "como leerlo: AUC 0,5 es no saber nada. Si el AUC por sede sube mucho por",
        "encima de 0,6 con rasgos tan pobres, hay que sospechar de la region antes",
        "que celebrar. Este numero es el piso que la red neuronal debe superar.",
    ]
    reporte = "\n".join(lineas)
    (RAIZ / "results" / "perfilado" / "03_linea_base.txt").write_text(reporte, encoding="utf-8")
    print("\n" + reporte)


if __name__ == "__main__":
    main()
