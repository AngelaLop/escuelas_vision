# Decisiones iniciales

23 de septiembre de 2026.

## Por qué un repositorio aparte

El trabajo de infraestructura escolar vive en `IDB/school_infrastructure` y es
una consultoría. Este proyecto reutiliza sus datos pero persigue otra cosa:
aprender el ciclo de producción de un modelo (registro de experimentos,
versiones, servicio, monitoreo, reentrenamiento). Mezclarlos ensuciaría los dos.

**Pendiente antes de publicar nada:** confirmar con el BID que un modelo
derivado de la encuesta del FFIE se puede mostrar como proyecto propio. El repo
hermano es público y sin restricciones, pero eso no se extiende solo a un
derivado.

## La etiqueta con la que se arranca

`techo_malo`, la binaria que ya usaba el proyecto hermano. Se sabe que es floja:

- La declara el rector, no un ingeniero.
- Concatena condiciones en un campo de texto.
- Viene sesgada hacia el daño, 77,4% de las fotos del conjunto de entrenamiento.

Se arranca con ella porque existe y permite montar el ciclo completo. La escala
ordinal al estilo de Yao et al. (ISVC 2025), anotada a mano con muestreo
balanceado, queda como fase posterior.

El script 01 deja además tres etiquetas separadas (agrietado, humedad, derruido)
y un conteo de condiciones, para poder preguntar después cuál de las tres es
visible en una fachada. La intuición dice que la humedad se ve y el
agrietamiento fino no, pero eso hay que medirlo.

## La partición

Por municipio, con semilla 20260923. Resultado del script 01:

| partición | fotos | sedes | municipios | techo malo |
|---|---|---|---|---|
| entrenamiento | 24.669 | 8.908 | 534 | 77,4% |
| validación | 5.250 | 1.890 | 120 | 83,6% |
| prueba | 6.047 | 2.184 | 119 | 77,4% |

Se excluyeron 1.381 fotos cuyo único valor declarado era "Otro, ¿Cuál?".

**Nota sobre el desbalance.** La validación quedó en 83,6% y las otras dos en
77,4%. Al agrupar por municipio no se puede estratificar libremente, así que esa
diferencia es el precio de no tener fuga. Importa al comparar métricas entre
particiones: hay que usar AUC y PR-AUC, no exactitud.

## La muestra que se descarga

El archivo completo son 35,8 GB y el servidor del FFIE responde en torno a 1,8
segundos por foto. Se descargan 1.200 sedes de entrenamiento, la mitad con techo
declarado malo y la mitad bueno, reduciendo el lado largo a 640 píxeles.

El muestreo es por sede y no por foto, porque la etiqueta es de la sede: partir
sus tres fotos entre particiones sería filtrar información.

## Qué sigue

1. Línea base con estadísticos de imagen (script 03), registrada en MLflow.
2. Ajuste de un modelo de visión preentrenado y comparación contra esa línea.
3. Empaquetado y servicio del modelo.
4. Monitoreo de deriva contra las fotos ciudadanas, que tienen otro encuadre.
