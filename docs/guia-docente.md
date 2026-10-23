# Guía para docentes: el interior de una neurona artificial

Cuatro actividades de 10–15 minutos para mostrar cómo «piensa» un modelo de lenguaje. No hace falta
saber de IA: cada actividad trae lo que hay que decir, lo que hay que mirar y qué **no** concluir.

## Antes de la clase

1. Instalá y arrancá la app ([README](../README.md#instalación)).
2. Generá una vez con **Qwen2.5-0.5B-Instruct** para que descargue el modelo (~1 GB). Sin conexión en el
   aula, esto tiene que estar hecho de antemano.
3. Dejá estos controles fijos:

| Control | Valor | Por qué |
|---|---|---|
| Modelo | Qwen2.5-0.5B-Instruct | Responde preguntas en español |
| Muestreo | **greedy** | Siempre elige lo más probable: el resultado se repite igual |
| **Rep. penalty** | **1** | El valor por defecto (1.15) cambia el texto y los ejemplos de esta guía dejan de coincidir |
| Tokens | 40 | Alcanza para una frase completa |

> Los textos de esta guía se verificaron con esa configuración. Con otro hardware o versión de las
> librerías pueden variar levemente: probá todo una vez antes de la clase.

## Vocabulario mínimo

| Palabra técnica | Decilo así |
|---|---|
| Token | Un trozo de palabra |
| Capa | Una etapa de procesamiento; la información pasa por 24 etapas en orden |
| Neurona | Un «interruptor» que se enciende más o menos según lo que lee |
| Cabeza de atención | Un ayudante que se fija en ciertas palabras del texto |
| Ablación | Apagar una neurona a propósito para ver qué cambia |

## Actividad 1 · El modelo escribe palabra por palabra (10 min)

**Prompt:** `¿Cuál es la capital de Francia?`

1. Pulsá **Generar** y pedí al grupo que mire el texto que aparece.
2. Señalá que no escribe la frase de una vez: elige **una palabra por vez**, mirando lo anterior.
3. Mirá el mapa de neuronas: «cada celda es una neurona; cada token las enciende distinto».

**Pregunta para el grupo:** ¿por qué creen que se encienden neuronas distintas con cada palabra?

**No concluir:** que cada neurona «sabe» una palabra. La información está repartida entre muchas.

## Actividad 2 · La señal atraviesa las capas (10 min)

**Prompt:** `Escribe una frase corta sobre el mar.`

1. Generá. En el panel de la izquierda, hacé clic en distintas capas (0, 10, 20, 23).
2. Comparen cómo cambia el patrón del mapa al avanzar.
3. Probá el selector **Forma** (cerebro/chip) para captar la atención del grupo y aclará que **es solo
   decorativo**.

**Pregunta:** si cada capa transforma lo que recibe, ¿qué esperan que pase en las últimas capas?

**No concluir:** que la zona del cerebro que se enciende «hace» algo. El lugar de cada neurona en el
dibujo es arbitrario.

## Actividad 3 · ¿En qué palabras se fija? (15 min)

**Prompt:** `¿Qué cosas te puedo preguntar?`

1. Generá y bajá a la sección **Atención**. Leé en voz alta la frase de **Lectura**.
2. Cambiá de ayudante (c0, c1, c2…) y leé cómo cambia la frase.
3. Mirá la grilla «Todos los ayudantes»: ¿hay filas muy concentradas en una sola palabra?

**Pregunta:** ¿por qué le serviría a un modelo fijarse en una palabra anterior para escribir la
siguiente?

**No concluir:** que haya un ayudante «bueno». Hay muchos y se complementan; algunos casi siempre miran
lo mismo.

## Actividad 4 · Apagar una neurona (15 min)

**Prompt:** `Escribe una frase corta sobre el mar.`

1. Generá. El texto base es:
   *«El mar es un espectáculo inestimable, una atmósfera de vida y movimiento que desafía la naturaleza
   y la humanidad.»*
2. Pedí una **predicción**: «si apago una neurona, ¿qué creen que pasa?».
3. En el overview, hacé clic en la **capa 20**. Elegí una neurona del ranking y pulsá **Ablar y
   regenerar**.

| Neurona (capa 20) | Resultado esperado |
|---|---|
| **#1586** | Cambio pequeño: «…que **captura** la naturaleza y la belleza del universo.» |
| **#3419** | Cambio medio: «El mar es un espectáculo de naturaleza y energía que es impresionante y fascinante.» |
| **#408** | Cambio grande: «El mar es un espacio infinito y lleno de vida, donde las estrellas brillan y los animales **vuelan**…» |

Mostrá **al menos dos**: una de efecto chico y otra de efecto grande. Es la mejor forma de transmitir
que no todas las neuronas pesan lo mismo.

**Pregunta:** ¿la neurona apagada «contenía» el mar? ¿Por qué el texto sigue siendo coherente?

**No concluir:**
- Que la neurona #408 «sea» las estrellas. Apagarla empuja el texto por otro camino, nada más.
- Que si el texto no cambia, la ablación falló. Muchas neuronas son redundantes: que no cambie también
  enseña algo (pasa con las neuronas #693 de la capa 8, por ejemplo).

## Qué no prometer

- **Mostrar actividad no es explicar significado.** Vemos números que suben y bajan, no ideas.
- **La forma cerebro/chip es decorativa.**
- **Es un modelo chico** (0,5 mil millones de parámetros). Los modelos grandes son mucho más complejos.
- **La ablación apaga una sola neurona**; no hay activation patching ni grupos de neuronas.

## Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| El texto no coincide con esta guía | Revisá que **Muestreo = greedy** y **Rep. penalty = 1** |
| La primera generación tarda mucho | Se está descargando el modelo; hacelo antes de la clase |
| La neurona que buscás no está en el ranking | El ranking muestra las 16 con mayor valor **en el último token**; usá las de la tabla, que se verificaron ahí |
| No aparece «Ablar y regenerar» activo | Primero hay que generar una vez sin ablación y seleccionar una neurona |

## Siguiente paso

Para grabar una demo corta, usá el [guion de video](./guion-video.md).
