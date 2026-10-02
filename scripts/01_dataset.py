"""Arma la base de fotos etiquetadas y fija la particion de entrenamiento.

Que hace. Lee el archivo de fotos que ya paso el triaje en el proyecto
school_infrastructure, deriva las etiquetas desde el campo declarado del techo y
reparte las sedes en entrenamiento, validacion y prueba agrupando por municipio.

Por que agrupa por municipio y no al azar. En school_infrastructure se midio que
un modelo entrenado solo con metadatos tecnicos de la foto (peso, dimensiones,
compresion) alcanza AUC 0,514 con particion aleatoria y 0,499 agrupando por
municipio. La diferencia es la firma del encuestador: el mismo funcionario
fotografio varias sedes del mismo municipio con la misma camara. Si una sede del
municipio queda en entrenamiento y otra en prueba, el modelo puede acertar
reconociendo la camara en vez de la cubierta. Agrupar por municipio cierra esa
puerta. Tambien cierra la de la sede, porque una sede esta en un solo municipio.

Las etiquetas que se derivan, y por que tres y no una:

  techo_malo      Binaria, la de school_infrastructure. Es con la que arranca el
                  proyecto porque ya existe y no cuesta nada.
  agrietado,      Multietiqueta. El campo declarado concatena condiciones
  humedad,        ("Agrietados Con exceso de humedad"), asi que cada condicion
  derruido        se puede leer por separado. Sirve para ver cual de las tres es
                  visible en una foto y cual no.
  n_condiciones   Conteo de 0 a 3. Es un sustituto pobre de la escala ordinal de
                  Yao et al. (1 a 5, siete evaluadores). No es lo mismo y no
                  pretende serlo: la escala ordinal hay que anotarla a mano mas
                  adelante. Se deja calculada para poder comparar despues.

Lo que este script NO hace. No descarga ninguna foto (eso es el script 02) y no
afirma nada sobre el estado real de un edificio. La etiqueta es lo que declaro
el rector en la encuesta del FFIE entre noviembre de 2021 y febrero de 2022, que
no es una medicion de ingenieria.

Una exclusion explicita. Las filas cuyo unico valor declarado es "Otro, Cual?"
quedan fuera: dicen que hay algo, no dicen que. Son 1.381 fotos.

Uso:
    uv run python scripts/01_dataset.py
    uv run python scripts/01_dataset.py --fuente <ruta a fotos_usables.csv>

Salidas:
    data/derived/fotos.parquet          una fila por foto, con etiquetas y particion
    results/perfilado/01_dataset.txt    el registro de lo que encontro
"""

import argparse
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAIZ = Path(__file__).resolve().parents[1]
FUENTE = RAIZ.parent / "IDB" / "school_infrastructure" / "results" / "fotos_usables.csv"

SEMILLA = 20260923
PROP_PRUEBA = 0.15
PROP_VALIDACION = 0.15


def normaliza(texto: str) -> str:
    """Minusculas sin tildes, para poder buscar dentro del campo declarado."""
    if not isinstance(texto, str):
        return ""
    sin_tilde = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_tilde.lower()


def etiquetas(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva las condiciones declaradas del campo de texto del techo."""
    declarado = df["D9Techos"].map(normaliza)

    df["buen_estado"] = declarado.str.contains("buen estado")
    df["agrietado"] = declarado.str.contains("agrietado")
    df["humedad"] = declarado.str.contains("humedad")
    df["derruido"] = declarado.str.contains("derruido")
    df["otro"] = declarado.str.contains("otro")

    df["n_condiciones"] = df[["agrietado", "humedad", "derruido"]].sum(axis=1)

    # Solo dice "Otro": hay algo pero no se sabe que. No sirve ni como positivo
    # ni como negativo, asi que sale.
    df["solo_otro"] = df["otro"] & (df["n_condiciones"] == 0) & ~df["buen_estado"]
    return df


def particion(df: pd.DataFrame) -> pd.DataFrame:
    """Reparte por municipio: primero prueba, y del resto sale validacion."""
    grupos = df["mpio_id"]

    corte1 = GroupShuffleSplit(n_splits=1, test_size=PROP_PRUEBA, random_state=SEMILLA)
    idx_resto, idx_prueba = next(corte1.split(df, groups=grupos))

    resto = df.iloc[idx_resto]
    prop_val = PROP_VALIDACION / (1 - PROP_PRUEBA)
    corte2 = GroupShuffleSplit(n_splits=1, test_size=prop_val, random_state=SEMILLA)
    idx_entrena, idx_val = next(corte2.split(resto, groups=resto["mpio_id"]))

    df["particion"] = "entrenamiento"
    df.iloc[idx_prueba, df.columns.get_loc("particion")] = "prueba"
    df.loc[resto.iloc[idx_val].index, "particion"] = "validacion"
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fuente", type=Path, default=FUENTE)
    args = ap.parse_args()

    if not args.fuente.exists():
        raise SystemExit(
            f"No encuentro {args.fuente}.\n"
            "Ese archivo lo produce el script 17 de school_infrastructure. "
            "Si el repositorio hermano esta en otra ruta, pasala con --fuente."
        )

    crudo = pd.read_csv(args.fuente)
    lineas = [f"fuente: {args.fuente}", f"fotos en la fuente: {len(crudo):,}"]

    df = etiquetas(crudo.copy())
    n_solo_otro = int(df["solo_otro"].sum())
    df = df[~df["solo_otro"]].copy()
    lineas.append(f"excluidas por decir solo 'Otro': {n_solo_otro:,}")

    # El municipio identifica el grupo. Se junta con el departamento porque hay
    # nombres de municipio repetidos entre departamentos.
    df["mpio_id"] = df["A5DeptoSed"].str.strip() + "|" + df["A6MpioSede"].str.strip()
    df = particion(df)

    columnas = [
        "CodDaneSed", "A5DeptoSed", "A6MpioSede", "mpio_id", "Zona",
        "url", "n_foto", "bytes", "ancho", "alto", "mpx", "bpp",
        "D9Techos", "techo_malo", "buen_estado",
        "agrietado", "humedad", "derruido", "n_condiciones",
        "particion",
    ]
    salida = df[columnas].rename(columns={
        "CodDaneSed": "cod_dane",
        "A5DeptoSed": "depto",
        "A6MpioSede": "mpio",
        "Zona": "zona",
    })

    destino = RAIZ / "data" / "derived" / "fotos.parquet"
    destino.parent.mkdir(parents=True, exist_ok=True)
    salida.to_parquet(destino, index=False)

    lineas += [
        "",
        f"fotos utilizables: {len(salida):,}",
        f"sedes: {salida.cod_dane.nunique():,}   municipios: {salida.mpio.nunique():,}",
        f"peso si se descargaran todas: {salida.bytes.sum() / 1e9:.1f} GB",
        "",
        "particion (fotos / sedes / municipios / % techo malo):",
    ]
    for nombre, g in salida.groupby("particion"):
        lineas.append(
            f"  {nombre:<14} {len(g):>7,}  {g.cod_dane.nunique():>6,}  "
            f"{g.mpio.nunique():>4,}   {g.techo_malo.mean():.1%}"
        )

    lineas += ["", "condiciones declaradas (share de fotos):"]
    for c in ["buen_estado", "agrietado", "humedad", "derruido"]:
        lineas.append(f"  {c:<12} {salida[c].mean():.1%}")

    lineas += ["", "control: ningun municipio puede estar en dos particiones"]
    cruce = salida.groupby("mpio_id")["particion"].nunique()
    lineas.append(f"  municipios en mas de una particion: {(cruce > 1).sum()}")

    reporte = "\n".join(lineas)
    perfil = RAIZ / "results" / "perfilado" / "01_dataset.txt"
    perfil.parent.mkdir(parents=True, exist_ok=True)
    perfil.write_text(reporte, encoding="utf-8")
    print(reporte)
    print(f"\nescrito: {destino}")


if __name__ == "__main__":
    main()
