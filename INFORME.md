# Predicción de costos médicos — Informe ejecutivo

**ML Ops Challenge · Resumen para decisión**

| | |
|---|---|
| **Qué se pidió** | Llevar un modelo de predicción de costos médicos de un script de laboratorio a algo que pueda operar en producción y avisar cuando algo anda mal |
| **Estado** | Funcionando de punta a punta, verificado |
| **Detalle técnico** | `informe.html` — informe completo, navegable |

---

## La conclusión, en tres líneas

1. **De los tres lotes de producción que se procesaron, dos traían datos defectuosos.**
   El sistema no sólo los detectó: dijo **cuál era el problema en cada uno**, que es lo
   que evita reaccionar mal.
2. **El modelo nuevo mejora poco** respecto del que ya existía: $34 de error promedio
   sobre pólizas de $13.270. El valor de este trabajo **no está en el modelo**.
3. **Está en que ahora se sabe cuándo confiar en él.** Antes no había forma de saberlo.

---

## El hallazgo central

Se procesaron tres lotes de 1.338 pólizas cada uno, sobre un padrón cuyo costo medio
facturado es de **$13.270** por póliza.

| Lote | Estado | Qué le pasaba |
|---|:---:|---|
| **1** | ✅ Sano | Nada. El formato del archivo era distinto, pero el dato era correcto |
| **2** | 🔴 Alerta | Los importes venían **multiplicados por 100** — como si el archivo estuviera en centavos |
| **3** | 🔴 Alerta | Un dato clínico (índice de masa corporal) venía **sin la coma decimal**, y el lote **no traía resultados reales** contra los cuales compararse |

Lo importante no es que los detectara. Es que **los distinguió**.

### Lote 2: el problema estaba en el dato, no en el modelo

El error de predicción se disparó a $1,79 millones. Con ese número sobre la mesa, la
reacción natural es reentrenar el modelo.

**Habría sido el error más caro posible**: se habría reentrenado sobre datos corruptos,
enseñándole al modelo a predecir centavos como si fueran dólares. El modelo de pricing
quedaría inutilizable, y el problema real —el proceso que genera el archivo— seguiría
ahí.

El sistema lo dice explícitamente:

> *Las características de los asegurados son estables y el desvío está sólo en los
> importes: apunta a un problema de calidad de datos en la etiqueta, no a degradación
> del modelo. Corregir el proceso que genera el archivo antes de considerar un
> reentrenamiento.*

### Lote 3: el caso que nadie habría visto

Este lote **no traía los costos reales**. Sin ellos, ninguna medida de error puede
delatar nada: el modelo predice, los números salen plausibles, y no hay con qué
contrastarlos.

Con el dato clínico corrupto, el modelo predijo un costo promedio de **$15.104** por
póliza, cuando su referencia histórica es **$12.529**.

> **Una sobreestimación sistemática del 20,6% sobre 1.338 pólizas: unos $3,4 millones
> de exposición mal tarifada** — que habría pasado como un resultado normal.

Se detectó por tres vías que no necesitan resultados reales: el dato viola los rangos
fisiológicamente posibles, la distribución de entrada se corrió respecto de lo conocido,
y la distribución de las predicciones también.

### Lote 1: la trampa silenciosa

Este lote estaba sano, pero usaba coma decimal en vez de punto. Leído con las
herramientas estándar **sin ninguna precaución**, el importe `14.700,80` se convierte en
`80.931` — sin error, sin aviso, sin nada que lo delate. Un archivo con la forma correcta
y los números equivocados.

El sistema lo lee bien, y hay una prueba automatizada que documenta el fallo del camino
ingenuo, para que nadie lo reintroduzca.

### Una decisión que conviene conocer

El sistema **detecta y alerta, pero no repara**. Sería técnicamente fácil corregir el
dato clínico del lote 3 y seguir. No se hace, por tres razones:

- El factor de corrección varía fila por fila: cualquier arreglo automático sería una
  conjetura.
- Reparar en silencio oculta el problema al proveedor de datos, que es donde hay que
  resolverlo.
- Un sistema que corrige entradas sin avisar produce decisiones que después nadie puede
  auditar.

Las predicciones igual se generan y se guardan. Lo que cambia es que llegan marcadas.

---

## El modelo: elegido midiendo, no opinando

El proyecto original usaba un solo algoritmo, y la justificación de esa elección era un
párrafo escrito a mano. Nunca se lo comparó contra nada.

Ahora cada entrenamiento prueba **cuatro enfoques distintos** sobre exactamente los
mismos datos, y gana el que mide mejor:

| Enfoque | Error promedio de predicción |
|---|---:|
| **Random Forest** ← el que quedó | **$4.800** |
| Gradient Boosting | $4.821 |
| El original (XGBoost) | $4.835 |
| Modelo lineal simple | $5.527 |

Dos lecturas de negocio:

**La mejora es marginal.** $34 sobre un error de $4.800, en pólizas de $13.270. No es un
argumento para cambiar nada por sí solo. Lo valioso es que ahora la elección **se puede
defender con un número** en vez de con una opinión, y que la próxima vez que alguien
proponga un modelo distinto habrá una forma objetiva de decidir.

**El modelo lineal simple queda $727 por detrás.** Eso sí importa: confirma que la
complejidad del modelo se está pagando con precisión real, y no por costumbre. Es la
clase de pregunta que conviene poder responder cuando alguien cuestione por qué el
sistema de pricing no es una fórmula.

---

## Los tres controles que ahora existen

### 1. Un modelo nuevo no entra a producción sin pasar pruebas

Entrenar y poner en producción son dos pasos separados a propósito. Un modelo recién
entrenado queda en espera; para reemplazar al que está sirviendo tiene que superar tres
condiciones objetivas, y **si no las supera, no entra**.

Ya pasó en la práctica: un candidato quedó rechazado por no mejorar al modelo vigente, y
el motivo quedó escrito. Sin esa separación, ese entrenamiento habría degradado el
sistema en silencio.

### 2. Cada predicción sabe de dónde salió

Cada estimación guardada registra qué modelo la produjo, qué versión y de qué
entrenamiento vino. Ante una consulta sobre una póliza puntual, se puede reconstruir
exactamente con qué se la calculó.

### 3. El monitoreo tiene memoria

Un reporte que dice lo mismo en cada corrida deja de leerse. Éste distingue tres cosas:

- **Nuevo** — apareció ahora. Es lo único que pide atención.
- **Persiste** — ya se sabía, lleva N corridas abierto.
- **Resuelto** — se arregló.

De modo que abrir el reporte y ver *«0 nuevas, 8 persisten»* es información útil: nada
cambió desde ayer. Y *«3 nuevas»* significa que hay algo que mirar hoy.

Los umbrales se pueden afinar **por dato y por lote**, sin tocar código. Un lote sin
resultados reales, por ejemplo, se mide con una vara más exigente en la entrada,
justamente porque no hay nada aguas abajo que pueda desmentirlo.

---

## Lo que este trabajo **no** cubre

Dicho explícitamente, para que no se asuma de más:

| Falta | Qué implica |
|---|---|
| **No avisa solo** | Las alertas quedan registradas y visibles, pero no salen por mail, Slack ni ningún canal. Alguien tiene que mirar el reporte |
| **No corre solo** | No hay tarea programada: el procesamiento se dispara a mano |
| **Los umbrales no están calibrados** | Son valores estándar de la industria, razonables, pero no ajustados con historia propia de la cartera |
| **Sin despliegue containerizado** | Se ejecuta en un entorno preparado a mano. La razón está explicada abajo |
| **Sin validación automática en cada cambio** | El código tiene 172 pruebas automatizadas, pero nadie las corre salvo que se las invoque |

### Sobre el despliegue containerizado

El proyecto original traía la configuración, y **se eliminó** en vez de adaptarse. El
motivo: el entorno de desarrollo no permitía ejecutarla, así que nunca se probó. Y ya
había señales de que estaba desactualizada respecto del resto de los cambios.

**Entregar infraestructura que nunca se ejecutó no es una garantía de reproducibilidad:
es una afirmación sin respaldo**, y si falla en manos de quien la recibe, es peor que su
ausencia. Todo lo que hacía se reduce a preparar la base de datos y ejecutar cinco
comandos en orden, que están documentados.

---

## Qué haría falta para operar de verdad

En orden de impacto:

1. **Un canal de aviso.** Es el paso más corto: el sistema ya distingue qué alerta es
   nueva, que es exactamente el evento que hay que notificar. Falta decidir a quién.
2. **Procesamiento programado.** Que corra solo con la frecuencia con la que lleguen los
   lotes.
3. **Calibrar los umbrales con historia real.** Los actuales son un punto de partida
   defendible; con unos meses de lotes se pueden ajustar a la cartera.
4. **Acordar con el proveedor de datos** un contrato de formato. Los tres defectos
   encontrados son de formato y escala, no de contenido: se evitan en origen.

---

## En una frase

El modelo mejoró poco. Lo que cambió es que **dos de cada tres lotes de producción
traían defectos que antes habrían pasado inadvertidos** —uno de ellos equivalente a
$3,4 millones de exposición mal tarifada— y ahora no sólo se detectan, sino que el
sistema explica cuál es el problema en cada caso.

---

## Dónde está el detalle

| Documento | Para quién |
|---|---|
| **`informe.html`** | El informe técnico completo: arquitectura, decisiones de diseño, verificación contra los criterios de aceptación e instrucciones de ejecución |
| **`README.md`** | Documento original del equipo de ciencia de datos, preservado sin modificar: la referencia sobre el modelo y el análisis exploratorio |
| **`challenge_ml.md`** | El enunciado del challenge |
