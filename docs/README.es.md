# Condición del techo escolar desde la foto de campo

¿Puede un modelo de visión leer en una foto de fachada lo que un rector declaró
sobre el techo de su escuela?

De las 52.823 sedes oficiales del país solo 15.244 fueron visitadas por la
encuesta del FFIE. El proyecto hermano `school_infrastructure` mostró que la
imagen satelital no distingue una sede en ruina de una en buen estado: el estado
de una escuela no se ve desde arriba. Lo que queda por probar es la foto de
campo, que sí mira la fachada de frente. Este repositorio es ese experimento, y
a la vez el ejercicio de llevarlo al ciclo completo de producción: registro de
experimentos, versiones del modelo, servicio, monitoreo y reentrenamiento.

## Qué hay y qué no

**Los datos** salen de `school_infrastructure`: 37.347 fotos que pasaron un
triaje de resolución y compresión, sobre 13.475 sedes. Ninguna foto vive en este
repositorio, todas se redescargan con el script 02.

**La etiqueta** es lo que declaró el rector en la encuesta del FFIE entre
noviembre de 2021 y febrero de 2022, en un campo de texto que concatena
condiciones: "Agrietados Con exceso de humedad". No es una medición de
ingeniería y viene sesgada hacia el daño, cerca de 75 a 25. El script 01 la
descompone en tres condiciones (agrietado, humedad, derruido) más la binaria
`techo_malo` que ya usaba el proyecto hermano.

**Lo que este proyecto no afirma.** Que una foto diga "techo malo" no significa
que el edificio esté en mal estado. Significa que la foto se parece a las fotos
de sedes cuyo rector declaró daño. La escala ordinal de verdad, al estilo de Yao
et al. (ISVC 2025), hay que anotarla a mano y todavía no existe.

## La regla que no se negocia

**Toda validación agrupa por municipio.** En `school_infrastructure` se midió que
los metadatos técnicos de la foto (peso, dimensiones, compresión) predicen el
techo declarado con AUC 0,514 en partición aleatoria y 0,499 agrupando por
municipio. Esa diferencia es la firma del encuestador, que fotografió varias
sedes vecinas con la misma cámara. Con partición aleatoria el desempeño sale
inflado y el modelo puede estar reconociendo la cámara en vez de la cubierta.

## Empezar

Se necesita [uv](https://docs.astral.sh/uv/).

```bash
uv run python scripts/01_dataset.py              # base y partición por municipio
uv run python scripts/02_descargar_fotos.py --sedes 1200
```

El script 01 espera encontrar el repositorio hermano en `../IDB/school_infrastructure`.
Si está en otra ruta, se pasa con `--fuente`.

## Estructura

```
data/raw/fotos/      las fotos descargadas, fuera de control de versiones
data/derived/        base de fotos con etiquetas y partición, manifiesto de descarga
scripts/             01 en adelante, en orden de ejecución
results/perfilado/   el registro en texto de lo que encontró cada script
docs/                decisiones y evidencia
```

## Fuentes

FFIE (encuesta de infraestructura y archivo fotográfico), vía el proyecto
`school_infrastructure`.
