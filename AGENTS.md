# AGENTS.md — Proyecto HateCR

## Objetivo general del proyecto

Construir notebooks reproducibles en Python para recolectar, limpiar, analizar y clasificar comentarios/respuestas en X relacionados con la campaña electoral 2025-2026 en Costa Rica.

El proyecto forma parte de un TFM de Humanidades Digitales titulado:

"Discursos de odio en redes sociales durante la campaña política 2025-2026 en Costa Rica"
Acrónimo: HateCR.

## Diseño metodológico

El corpus formal analiza comentarios en X publicados en torno a seis momentos cronologicos de la campana electoral costarricense 2025-2026.

La unidad de análisis principal es el comentario/respuesta en X.

La unidad de recolección es la publicación fuente o post madre publicada por una cuenta activa de un medio costarricense configurado en `config/media_accounts.yaml`.

Alcance de medios:
- Usar todos los medios con `active: true` en `config/media_accounts.yaml`.
- No mantener listas rigidas de tres medios dentro de notebooks o scripts.
- Toda query de posts madre debe contener `from:{media_handle}`.

Eventos formales:
1. Inicio formal de campana — 1 de octubre de 2025.
2. Primer encuentro/debate del TSE — 9 de enero de 2026.
3. Cierre de encuentros del TSE — 12 de enero de 2026.
4. Debate Repretel–Radio Monumental — 27 de enero de 2026.
5. Cierre definitivo de campana — 28 de enero de 2026.
6. Jornada electoral — 1 de febrero de 2026.

## Reglas de recolección

Usar únicamente métodos compatibles con acceso autorizado a datos de X.

No implementar evasión de bloqueos, scraping agresivo, automatización de navegador para eludir restricciones, rotación de proxies ni extracción no autorizada.

Las credenciales deben leerse desde un archivo `.env`.

Nunca guardar tokens, API keys ni secretos dentro de notebooks, scripts o commits.

La recoleccion formal de replies debe prepararse primero en modo `dry-run`. Una ejecucion real debe requerir una bandera explicita, una confirmacion y un batch identificado; ejecutar todas las celdas no debe llamar accidentalmente a la API.

Guardar datos crudos en `data/raw/`.

Guardar datos limpios en `data/processed/`.

## Variables mínimas esperadas

Para cada post madre:
- event_id
- event_name
- media_account
- source_post_id
- source_post_text
- source_post_created_at
- source_post_url
- source_post_public_metrics

Para cada comentario/respuesta:
- event_id
- media_account
- source_post_id
- reply_id
- reply_text
- reply_created_at
- reply_author_id_hash
- lang
- public_metrics
- conversation_id
- in_reply_to_user_id
- referenced_tweets
- collected_at

## Protección ética

No guardar nombres reales de usuarios comunes si no es necesario.

Crear una columna con hash del author_id.

No publicar comentarios textuales altamente violentos sin anonimización.

Separar análisis agregado de ejemplos textuales.

## Etiquetado manual formal

El archivo canonico de anotacion es:

`reports/formal_eda/manual_review_sample.csv`

Reglas de proteccion:
- No sobrescribir este archivo si ya existe.
- Los notebooks 05 y 06 deben leerlo, validarlo y combinarlo por `tweet_id`, pero nunca escribir sobre el.
- Cualquier nueva muestra generada por el notebook 04 debe guardarse como `manual_review_sample_candidate.csv`.
- La muestra esta enriquecida por lexicon; esta seleccion debe declararse al interpretar metricas.

Contrato de etiquetado humano vigente:
- `hostility_relevance`: nivel `0`, `1`, `2` o `3`.
- `manual_hostility`: binaria; `0` no se encontro hostilidad/incivilidad y `1` si se encontro.
- `manual_hate_speech`: binaria; `0` no se encontro odio y `1` si se encontro.
- `notes`: contexto, duda o justificacion opcional.
- Nivel `0`: no ofensivo ni odio.
- Nivel `1`: incivilidad o descalificacion sin violencia ni ataque identitario.
- Nivel `2`: deseo, amenaza, aprobacion o llamado al dano sin fundamento identitario.
- Nivel `3`: ataque contra personas o grupos por identidad, cultura o adscripcion ideologica.

Los niveles forman una taxonomia jerarquica, no una escala pura de intensidad. El
nivel 3 tiene prioridad cuando el ataque se basa en identidad, cultura o ideologia.
Criticar una idea o partido no basta: el ataque debe recaer sobre personas por esa
adscripcion.

Patron orientativo para diagnostico:
- Nivel `0`: `manual_hostility=0`, `manual_hate_speech=0`.
- Nivel `1`: `manual_hostility=1`, `manual_hate_speech=0`.
- Nivel `2`: `manual_hostility=1`, `manual_hate_speech=0`.
- Nivel `3`: `manual_hostility=1`, `manual_hate_speech=1`.

Las diferencias con este patron generan advertencias, no errores. Las etiquetas
binarias son decisiones humanas independientes y no deben sobrescribirse ni excluirse
automaticamente para forzar la correspondencia con el nivel.

Hostilidad e incivilidad no equivalen a odio. `src/labels.py` conserva las decisiones
humanas, crea `y_hostility` desde `manual_hostility`, crea `y_hate_speech` desde
`manual_hate_speech` y usa el nivel para el target multiclase y los controles de
consistencia. Nunca debe completar o sobrescribir automaticamente las etiquetas.

## Estilo de código

Usar Python 3.11 o superior.

Usar funciones reutilizables dentro de `src/`.

Los notebooks deben llamar funciones desde `src/` en vez de repetir demasiado código.

Usar pandas, requests, python-dotenv, pyyaml, tqdm, matplotlib, scikit-learn, spaCy o librerías equivalentes.

Cada notebook debe incluir:
1. Objetivo.
2. Entradas.
3. Salidas.
4. Código.
5. Validaciones.
6. Breve interpretación metodológica.

## Done when

El proyecto estará completo cuando existan notebooks ejecutables que:

1. Lean eventos y cuentas desde archivos YAML.
2. Se conecten a la API de X mediante variables de entorno.
3. Identifiquen posts madre por cuenta, fecha y términos de búsqueda.
4. Recolecten respuestas/comentarios asociados a cada post madre.
5. Guarden CSV/Parquet reproducibles.
6. Limpien y anonimicen los datos.
7. Generen tablas y gráficos iniciales.
8. Dejen preparado un dataset para análisis de sentimiento y clasificación de odio.
