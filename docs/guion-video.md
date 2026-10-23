# Guion de video: el interior de una neurona artificial (3 min)

Un video corto, sin jerga, que muestra cómo un modelo escribe y qué pasa si se le apaga una neurona. El
prompt y las neuronas de este guion se probaron con el modelo real.

## Preparación (5 min, antes de grabar)

| Paso | Detalle |
|---|---|
| Modelo | Qwen2.5-0.5B-Instruct, ya descargado (primera generación lenta) |
| Muestreo | **greedy** |
| **Rep. penalty** | **1** (con 1.15 el texto cambia y el guion no coincide) |
| Tokens | 40 |
| Forma | Cerebro (más vistoso) |
| Pantalla | Navegador a 1400 px de ancho o más; zoom 100 % |
| Ensayo | Corré todo el guion una vez entero antes de grabar |

Texto base esperado (si no coincide, revisá la tabla):

> *El mar es un espectáculo inestimable, una atmósfera de vida y movimiento que desafía la naturaleza y
> la humanidad.*

## Guion

| Tiempo | En pantalla | Qué decir |
|---|---|---|
| 0:00 | Título de la app | «Un modelo de lenguaje escribe una palabra por vez. Vamos a ver qué pasa por dentro.» |
| 0:15 | Escribir el prompt `Escribe una frase corta sobre el mar.` y pulsar **Generar** | «Le pedimos una frase sobre el mar.» |
| 0:25 | Aparece el texto y se encienden las neuronas del cerebro | «Cada celda es una neurona. Se encienden más o menos según la palabra que está escribiendo. El dibujo del cerebro es solo decorativo.» |
| 0:55 | Clic en capas 0, 10 y 23 del panel izquierdo | «La información pasa por 24 etapas, que llamamos capas. En cada una el patrón cambia.» |
| 1:20 | Bajar a **Atención** y leer la frase de **Lectura** | «Antes de escribir, el modelo repasa lo ya escrito y se fija más en unas palabras que en otras. Cada ayudante mira cosas distintas.» |
| 1:50 | Clic en la **capa 20** y luego en la neurona **#1586** → **Ablar y regenerar** | «Ahora apagamos una sola neurona, de entre casi cinco mil.» |
| 2:05 | Mostrar el texto nuevo («…que captura la naturaleza y la belleza del universo.») | «Cambió el final de la frase, pero sigue siendo parecida.» |
| 2:15 | Clic en la neurona **#408** → **Ablar y regenerar** | «Probemos con otra.» |
| 2:30 | Mostrar el texto nuevo («…donde las estrellas brillan y los animales vuelan…») | «Esta vez la frase se fue por otro camino. No todas las neuronas pesan lo mismo.» |
| 2:50 | Plano general | «No sabemos qué «significa» cada neurona. Pero podemos ver cómo se enciende, y qué pasa si la apagamos.» |

## Frases para evitar

| No digas | Mejor |
|---|---|
| «Esta neurona es la del mar» | «Apagarla cambia el rumbo de la frase» |
| «El modelo piensa» | «El modelo calcula la siguiente palabra» |
| «Esta zona del cerebro hace X» | «La forma es solo un dibujo» |

## Si algo sale distinto

- **El texto base no coincide:** revisá greedy y Rep. penalty = 1.
- **La neurona no está en el ranking:** el ranking refleja el último token; seleccioná la capa 20 *después*
  de que termine la generación.
- **La ablación no cambia el texto:** no falló; usá la #408, que produjo el cambio más grande.

## Para grabar

- Grabá la pantalla en 1080p con la app a pantalla completa y sin notificaciones.
- Narrá por separado y montá después: es más fácil corregir una frase que repetir toda la toma.
- Dejá 1–2 s de pausa tras cada generación para que se vea el resultado.
- Subtitulá en español: ayuda a quien lo ve sin sonido.

## Siguiente paso

Para llevarlo al aula, usá la [guía para docentes](./guia-docente.md).
