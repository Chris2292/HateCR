# Disponibilidad y gobernanza de datos

## Alcance de esta versión pública

Este repositorio distribuye código, configuración metodológica, notebooks sin salidas y resultados exclusivamente agregados. No distribuye el corpus hidratado obtenido mediante la API de X.

No se incluyen:

- textos completos de posts, replies o quote posts;
- usernames, handles de usuarios comunes o identificadores de usuario;
- Post IDs o conversation IDs;
- hashes de autores;
- datos crudos, intermedios o procesados;
- archivos de etiquetado manual;
- predicciones a nivel de comentario;
- modelos serializados entrenados con el corpus;
- lexicones externos cuya licencia no haya sido verificada.

## Resultados públicos

`docs/showcase/` contiene únicamente tablas agregadas y figuras que no permiten reconstruir comentarios individuales. Algunas tablas mencionan candidatos, partidos, instituciones o medios en su condición de actores públicos del proceso electoral.

## Reproducción de la recolección

La reproducción completa requiere:

1. acceso propio y autorizado a la API de X;
2. aceptación y cumplimiento de los términos vigentes de X;
3. credenciales almacenadas localmente en `.env`;
4. una sal aleatoria propia para anonimizar autores;
5. revisión ética e institucional adecuada al contexto de uso;
6. disponibilidad de los endpoints y niveles de acceso necesarios.

El repositorio no proporciona credenciales compartidas ni mecanismos alternativos de acceso. No implementa scraping, Selenium, proxies, rotación de identidades ni evasión de límites.

## Uso responsable

Los resultados agregados no deben utilizarse para inferir atributos sensibles de usuarios individuales, vigilar personas, construir perfiles personales ni identificar autores anónimos. Las predicciones de hostilidad y odio son exploratorias y no sustituyen revisión humana contextual.

## Política de X

Antes de reutilizar el proyecto, consulte las versiones vigentes de:

- https://docs.x.com/developer-terms/policy
- https://docs.x.com/developer-terms/restricted-use-cases
- https://docs.x.com/developer-terms/agreement

Las condiciones de acceso y redistribución pueden cambiar. La persona que reutilice el código es responsable de verificar su autorización y obligaciones aplicables.
