# Estado del plan de mejoras

El plan original está completado para el uso previsto: una persona ejecuta el visualizador localmente,
observa activaciones, compara pasadas y puede ablacionar una neurona. No hay trabajo obligatorio
pendiente. La escala visual por capa se añadió como opción, conservando la escala global como valor
predeterminado.

## Estado

| Área | Estado | Resultado |
|---|---|---|
| Fiabilidad del WebSocket | Hecho | Validación de mensajes, errores legibles, recuperación de la UI y decodificación incremental. |
| Modelo y generaciones | Hecho | Carga protegida por lock; las generaciones se serializan para compartir un solo modelo sin mezclar estado. |
| Rendimiento | Hecho | KV cache, reutilización del prefill, caché limitado a la última posición y payload reducido a la capa visible. |
| Interfaz y atención | Hecho | Cambio de capa, atención top-8 explicada, canvas limitado a ~30 fps y soporte retina. |
| Muestreo e informe | Hecho | Controles de muestreo completos e informe con semántica documentada. |
| Comparación, ablación y exportación | Hecho | Comparación A/B, ablación de una neurona con semilla reproducible, CSV y Parquet. |
| Empaquetado y documentación | Hecho | Docker CPU, dependencias fijadas y requisitos de memoria documentados. |
| Calidad continua | Hecho | Tests unitarios y de integración, lint y CI. |
| Atención explicada | Hecho | Introducción en lenguaje llano, frase «Lectura» dinámica, etiquetas sin jerga y ayudas **?**. |
| Forma del mapa | Hecho | Selector cuadrícula / cerebro / chip; decorativo, la cuadrícula sigue por defecto. |
| Ayuda contextual | Hecho | Iconos **?** con popover explicativo en cada control de muestreo y entrada. |
| Escala visual por capa | Hecho | Opción para ajustar el contraste a la capa visible; la escala global sigue disponible. |

Los detalles de implementación y uso están en [NEXT_STEPS.md](./NEXT_STEPS.md) y [README.md](./README.md).

## Alcance deliberado

Estas capacidades no son requisitos para el uso local previsto y no forman parte de tareas pendientes:

- **Concurrencia real:** las generaciones se ejecutan en serie; un segundo cliente espera en cola. No se
  crearán instancias paralelas del modelo.
- **Más tipos de intervención:** la ablación se limita a una neurona por generación. No se planifica
  ablación de grupos ni activation patching.
- **Más datos de atención o contexto:** se mantiene el top-8 por cabeza y el límite de 512 tokens de entrada.
- **Otras plataformas o infraestructura:** no se planifican migración a nnsight/NDIF ni variante Docker GPU.
- **Compresión de Parquet:** se mantiene el escritor actual sin compresión ni dependencias adicionales.
- **Más pruebas de interfaz:** las pruebas UI adicionales para comparación, exportación, reconexión y
  cambio de capa no son necesarias para el alcance actual.

## Decisiones de visualización

- La **escala global** es el valor predeterminado y permite comparar magnitudes entre capas.
- **Escala por capa** es una opción visual para distinguir mejor activaciones pequeñas cuando otra capa
  tiene un valor atípico. Al activarla, los colores de capas distintas dejan de ser comparables; los
  valores numéricos, rankings, informes y exportaciones no cambian.
- En el mapa diferencial se conserva la escala propia de `máx |Δ|`.

## Verificación y cierre

- La suite configurada cubre sampling, validación, decodificación, generación, estado de sesión,
  endpoint WebSocket y salud; las pruebas de integración con modelo real y de navegador tienen sus
  marcadores correspondientes.
- La prueba de navegador para la escala por capa está añadida en
  `tests/test_frontend_ablation.py`. En el entorno donde se incorporó, Playwright no estaba disponible,
  así que esa prueba se omitió; el comportamiento sí se comprobó manualmente en navegador.
- No hay siguiente incremento planificado. Reabrir el plan solo si aparece una necesidad concreta de uso
  que justifique ampliar este alcance.
