"""Descarga una muestra de fotos del archivo del FFIE y las deja listas para entrenar.

Por que una muestra y no todo. El archivo completo son 37 GB y cada foto tarda
cerca de 1,8 segundos en llegar. Bajarlo entero seria mas de un dia de descarga
para una linea base que no lo necesita. El script baja sedes completas, no fotos
sueltas, porque la etiqueta es de la sede: partir sus fotos entre particiones
seria filtrar informacion.

Por que se reduce la imagen al guardar. Se guarda el lado largo en 640 pixeles.
Los modelos de vision que vamos a usar reciben 224, asi que 640 deja margen para
recortes y aumentos sin guardar 37 GB. El triaje de school_infrastructure ya
descarto lo que venia por debajo de medio megapixel, de modo que ninguna foto
util se pierde por este corte. El peso original queda registrado en el manifiesto.

Por que es reanudable. La descarga se corta: se cae la red, se cierra el portatil.
Una foto ya guardada no se vuelve a pedir nunca, y el manifiesto se reescribe en
cada corrida con lo que hay en disco. Se puede interrumpir con Ctrl+C sin dañar
nada.

Cortesia con el servidor. geodata.ffie.com.co es un servidor publico de una
entidad, no una CDN. Se usan pocas conexiones en paralelo y hay una pausa entre
peticiones. No subas mucho el paralelismo.

Uso:
    uv run python scripts/02_descargar_fotos.py --sedes 1200
    uv run python scripts/02_descargar_fotos.py --sedes 1200 --hilos 6
    uv run python scripts/02_descargar_fotos.py --particion validacion --sedes 300

Salidas:
    data/raw/fotos/<cod_dane>_<n>.jpg
    data/derived/fotos_descargadas.csv   manifiesto: que se pidio y como salio
"""

import argparse
import io
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "data" / "derived" / "fotos.parquet"
DESTINO = RAIZ / "data" / "raw" / "fotos"
MANIFIESTO = RAIZ / "data" / "derived" / "fotos_descargadas.csv"

LADO_LARGO = 640
CALIDAD = 88
ESPERA = 0.05          # segundos entre peticiones de un mismo hilo
INTENTOS = 3
TIEMPO_LIMITE = 30
SEMILLA = 20260923

# Una sesion por hilo. Sin esto cada foto abre una conexion TLS nueva, y el
# apreton de manos con el servidor del FFIE cuesta mas que la descarga misma:
# medido, 16 fotos por minuto con conexiones sueltas.
_local = threading.local()


def sesion() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "escuelas-vision/0.1 (investigacion academica)"
        adaptador = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        s.mount("https://", adaptador)
        _local.s = s
    return _local.s


def ruta_local(cod_dane: int, n_foto: int) -> Path:
    return DESTINO / f"{cod_dane}_{n_foto}.jpg"


def baja_una(fila: pd.Series) -> dict:
    """Pide una foto, la reduce y la guarda. Devuelve la fila del manifiesto."""
    destino = ruta_local(fila.cod_dane, fila.n_foto)
    registro = {
        "cod_dane": fila.cod_dane,
        "n_foto": fila.n_foto,
        "url": fila.url,
        "archivo": str(destino.relative_to(RAIZ)),
        "bytes_origen": fila.bytes,
    }

    if destino.exists():
        return {**registro, "estado": "ya_estaba", "bytes_guardados": destino.stat().st_size}

    for intento in range(1, INTENTOS + 1):
        try:
            r = sesion().get(fila.url, timeout=TIEMPO_LIMITE)
            if r.status_code != 200:
                return {**registro, "estado": f"http_{r.status_code}", "bytes_guardados": 0}

            img = Image.open(io.BytesIO(r.content))
            img = img.convert("RGB")
            img.thumbnail((LADO_LARGO, LADO_LARGO), Image.LANCZOS)
            img.save(destino, "JPEG", quality=CALIDAD, optimize=True)

            time.sleep(ESPERA)
            return {**registro, "estado": "ok", "bytes_guardados": destino.stat().st_size}

        except Exception as e:                      # noqa: BLE001
            if intento == INTENTOS:
                return {**registro, "estado": f"error: {type(e).__name__}", "bytes_guardados": 0}
            time.sleep(2 * intento)

    return {**registro, "estado": "sin_resolver", "bytes_guardados": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sedes", type=int, default=1200, help="cuantas sedes bajar")
    ap.add_argument("--particion", default="entrenamiento")
    ap.add_argument("--hilos", type=int, default=6)
    args = ap.parse_args()

    if not BASE.exists():
        raise SystemExit("Falta data/derived/fotos.parquet. Corre antes el script 01.")

    base = pd.read_parquet(BASE)
    sub = base[base.particion == args.particion]

    # Muestreo de sedes, no de fotos, y balanceado entre techo malo y bueno para
    # que la linea base no arranque con 75 a 25 en contra.
    por_sede = sub.groupby("cod_dane").techo_malo.max()
    rng = random.Random(SEMILLA)
    malas = [c for c, v in por_sede.items() if v == 1]
    buenas = [c for c, v in por_sede.items() if v == 0]
    rng.shuffle(malas)
    rng.shuffle(buenas)
    mitad = args.sedes // 2
    elegidas = set(malas[:mitad] + buenas[: args.sedes - mitad])

    tareas = sub[sub.cod_dane.isin(elegidas)].copy()
    print(
        f"particion {args.particion}: {len(elegidas):,} sedes, {len(tareas):,} fotos "
        f"({tareas.techo_malo.mean():.0%} techo malo)"
    )
    DESTINO.mkdir(parents=True, exist_ok=True)

    filas, hechas = [], 0
    try:
        with ThreadPoolExecutor(max_workers=args.hilos) as pool:
            for r in pool.map(baja_una, [f for _, f in tareas.iterrows()]):
                filas.append(r)
                hechas += 1
                if hechas % 50 == 0:
                    ok = sum(1 for x in filas if x["estado"] in ("ok", "ya_estaba"))
                    print(f"  {hechas:,}/{len(tareas):,}  ok {ok:,}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrumpido: lo descargado se conserva y el manifiesto se guarda")

    man = pd.DataFrame(filas)
    if MANIFIESTO.exists():
        previo = pd.read_csv(MANIFIESTO)
        man = pd.concat([previo, man]).drop_duplicates(["cod_dane", "n_foto"], keep="last")
    man.to_csv(MANIFIESTO, index=False)

    print("\nestado de la descarga:")
    print(man.estado.value_counts().to_string())
    guardado = man.bytes_guardados.sum() / 1e6
    print(f"\nen disco: {guardado:,.0f} MB   manifiesto: {MANIFIESTO}")


if __name__ == "__main__":
    main()
