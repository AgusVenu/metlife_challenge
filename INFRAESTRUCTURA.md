# Infraestructura — qué sería containerizar esto

## 1. Cómo se repartiría el trabajo

Serían **tres contenedores**, y la división no es arbitraria: separa lo que tiene estado
de lo que no, y lo que hace el trabajo de lo que sólo lo muestra.

### El que guarda (`postgres`)

Es **el único con estado**, y adentro conviven **dos bases separadas**:

- **la del negocio** — el padrón de entrenamiento, las predicciones de cada lote, el
  monitoreo y el historial de alertas;
- **la de MLflow** — experimentos, métricas, y el registro de qué modelo estuvo en
  producción y desde cuándo.

Van separadas porque MLflow crea 59 tablas propias: mezclarlas con las del negocio vuelve
ilegible el esquema de la aplicación.

**Es el contenedor que importa.** Si se pierde su volumen se pierde la memoria del
sistema: qué modelo sirvió cuándo, y qué alertas estaban abiertas. El monitoreo deja de
poder decir «esto ya lo sabíamos».

### El que trabaja (`ml_pipeline`)

Arranca, corre las siete etapas —pruebas, preparar la base, entrenar, promover, puntuar—
y **termina**. No es un servicio que queda escuchando: es un trabajo por lotes.

Esa distinción tiene una consecuencia práctica. El proyecto original lo configuraba para
reintentar ante un fallo, que es lo correcto para un servicio caído y lo peor posible
para un trabajo por lotes: un pipeline que falla por un dato mal formado se reintentaría
en bucle. Debe fallar **una vez** y dejar el registro.

### El que muestra (`mlflow_ui`)

La interfaz de MLflow, apuntando a la misma base. No participa del procesamiento: se
levanta aparte cuando alguien quiere mirar, y el pipeline corre igual sin él.

### Lo que **no** entra en ninguna imagen

Dos cosas se montan desde afuera, y por motivos distintos:

- **Los datos de producción.** Los entrega un proveedor y cambian sin que cambie el
  código. Si estuvieran dentro de la imagen, agregar un lote nuevo obligaría a
  reconstruirla. Además se montan de **sólo lectura**, coherente con que el pipeline
  detecta y alerta, pero no repara.
- **Los modelos y sus artefactos.** Tienen que sobrevivir a que se baje el contenedor.
  Entre ellos viaja el *baseline* contra el que se mide el drift: si se pierde, el
  monitoreo pierde su punto de comparación.

---

## 2. Qué se gana y qué se paga

| | Hoy (local) | Con contenedores |
|---|---|---|
| **Arrancar de cero** | Instalar Python 3.11, PostgreSQL, crear dos bases, armar el entorno | Un comando |
| **«En mi máquina anda»** | Depende del sistema operativo y de las versiones instaladas | Reproducible |
| **Convivencia** | El PostgreSQL del proyecto comparte la máquina con todo lo demás | Aislado |
| **Iterar en el código** | Inmediato | Reconstruir la imagen ante cada cambio |
| **Depurar** | Directo | Con fricción |
| **Costo de entrada** | Cero: ya está hecho y verificado | Instalar Docker **y verificar que corre** |
| **Manejo de secretos** | Archivo local | Igual. **No mejora** |
| **Trabajo en equipo** | No escala | Tampoco, sin los pasos de § 4 |

La lectura corta: **containerizar resuelve reproducibilidad, no productivización.** Es un
paso real, pero más chico de lo que suele suponerse.

---

## 3. Lo que containerizar **no** resuelve

Cinco huecos que seguirían exactamente igual:

**1. Todo el mundo necesita la contraseña de la base.** Hoy cada proceso que registra algo
en MLflow se conecta directo a PostgreSQL, así que necesita sus credenciales. Por eso hay
un script dedicado sólo a levantar la interfaz sin imprimirla. Lo correcto es un
**servidor de tracking** delante: los clientes le hablan a él, y las credenciales quedan
de un solo lado. **El código ya lo soporta** — es cambiar una dirección en la
configuración.

**2. Los modelos viven en un disco.** No sobreviven a otra máquina. Para un equipo hace
falta un almacenamiento compartido en la nube.

**3. Nadie dispara el proceso.** No hay tarea programada: alguien lo ejecuta a mano. El
pipeline ya está listo para que lo haga un programador de tareas —cada etapa devuelve un
código de salida, y se puede configurar para que termine en error si algún lote queda en
alerta— pero falta decidir cada cuánto.

**4. Nadie valida los cambios automáticamente.** Las 172 pruebas del proyecto **no
necesitan base de datos ni MLflow**, se diseñaron así a propósito: se pueden correr en
menos de un minuto en cualquier servicio de integración continua, sin levantar nada.

**5. Las alertas no salen a ningún lado.** Quedan registradas y visibles, pero nadie se
entera salvo que mire. El enganche existe —el sistema ya distingue qué alerta es nueva,
que es exactamente lo que hay que notificar— y lo que falta es decidir a quién.

---

## 4. Las cuatro opciones, y cuál elegiría

| Opción | Cuesta | Da |
|---|---|---|
| **Seguir como está** | Nada | Funciona, pero cada persona nueva pierde un rato armando el entorno |
| **Sólo la base en un contenedor** | Minutos | Elimina la parte molesta del setup y el conflicto con otros PostgreSQL, sin perder la inmediatez de iterar en el código |
| **Todo containerizado** | Horas, más verificarlo | Reproducibilidad completa. Es lo que describe este documento |
| **Orquestador (Kubernetes y afines)** | Semanas | Desproporcionado: esto procesa tres lotes por corrida |

**La segunda es la mejor relación costo/beneficio para desarrollo**, y suele pasarse por
alto: da casi todo el beneficio práctico —setup reproducible, aislamiento— sin el costo de
reconstruir una imagen en cada cambio de código.

La tercera se justifica cuando el pipeline tenga que correr en un servidor que no es el de
nadie.

---

## 5. Si se hace, qué hay que corregir

Los archivos originales siguen en el historial (`git show decb3a1:docker-compose.yaml`),
pero **restaurarlos no alcanza**: el pipeline cambió bastante desde entonces.

| Cambió | Qué implica |
|---|---|
| MLflow pasó a guardar en PostgreSQL | Hace falta crear la **segunda base** al inicializar el contenedor |
| Se unificaron los experimentos | El archivo viejo todavía los partía en dos. **Hay que borrar esas variables**, no adaptarlas |
| Apareció el paso de promoción | Va entre entrenar y puntuar, y **no** debe abortar el pipeline si rechaza al candidato: un rechazo es una decisión válida |
| Apareció el archivo de reglas de monitoreo | Hay que incluirlo en la imagen |
| Apareció el historial de alertas | El volumen de la base pasa a ser lo que le da memoria al sistema |
| Se subió a Python 3.11 | El archivo original apuntaba a 3.10 |

Esa segunda fila es la que más importa, y es la que justifica todo este documento: **el
archivo viejo separaba los experimentos de MLflow después de que la solución los
unificara**, de modo que lo que la documentación promete sobre comparar validación contra
producción en un mismo gráfico **no se cumplía dentro del contenedor**. Es el ejemplo
exacto de por qué se eliminó en lugar de adaptarse a ojo.

### Cómo saber que quedó bien

Razonar no es verificar. Antes de dar esto por bueno hay que ver: que la imagen construya,
que las dos bases se creen solas, que el pipeline complete las siete etapas, que la
interfaz de MLflow muestre **un solo** experimento, y que **una segunda corrida seguida no
repita las alertas** — si algo aparece como «nueva» dos veces, la deduplicación está rota.

Recién ahí esto deja de ser un diseño.
