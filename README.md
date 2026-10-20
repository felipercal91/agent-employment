# Job Search Assistant

Asistente multiagente personal para la búsqueda de empleo (Senior Java Backend, Toronto / Montreal / remoto Canadá). **Automatiza la preparación, nunca el envío**: todo lo que genera queda pendiente de tu aprobación. Ver reglas en [CLAUDE.md](CLAUDE.md).

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # luego edita .env y pon tu ANTHROPIC_API_KEY
```

## Verificar

```powershell
python -m pytest                 # tests locales, no llaman a la API
python -m job_assistant check    # verifica .env y hace una llamada mínima a Claude
```

## Profile / RAG (Fase 2)

```powershell
python -m job_assistant index                                   # indexa data/ (re-ejecutar al cambiar documentos)
python -m job_assistant ask "Have I used Kafka?"                # pregunta sobre tu perfil
python -m job_assistant ask "What did interviewers criticize?" --category feedback
```

Las respuestas se basan solo en tus documentos y citan las fuentes; si algo no está, responde `Not found in your documents.`
## Fit + Requirements Check (Fase 3)

Requiere `data/profile.toml` (copia de `examples/profile.example.toml` con tus datos reales).

```powershell
python -m job_assistant evaluate examples/postings/senior_java_toronto.txt
python -m job_assistant evaluate examples/postings/product_owner_cpo.txt   # termina en STOP por CPO
```

- **Fit score**: determinista (sin LLM). % de keywords técnicas del posting cubiertas por tus `skills`, −15 si piden más años de los que tienes. Muestra las que faltan.
- **Requirements Check**: Claude clasifica cada requisito (must-have / nice-to-have; met / not_met / unclear). Si algún must-have es `not_met`, el flujo se detiene (exit code 3) y no se prepara material.

## Tracker de Excel (Fase 4)

Por defecto usa `data/tracker.xlsx` (trabaja sobre una **copia** de tu tracker). Detecta las columnas por nombre en cualquier hoja (Empresa/Company, Puesto/Role, Link/URL, Fit %, Estado/Status, Fecha/Date, Notas/Notes).

```powershell
python -m job_assistant tracker list                                     # hojas, columnas detectadas y filas
python -m job_assistant tracker check --company "Shopify" --role "Backend Engineer"
python -m job_assistant evaluate posting.txt --company "Acme" --link "https://..." --add
```

- Duplicado = mismo link (ignorando parámetros de tracking) o misma empresa + mismo puesto. Si es duplicado, no se llama a Claude.
- `--add` solo agrega si pasó el Requirements Check. Antes de escribir crea un backup en `data/backups/` y se niega si el archivo está abierto en Excel.

## Tailoring + Quebec (Fase 5)

```powershell
python -m job_assistant tailor posting.txt --company "Acme"            # idioma automático
python -m job_assistant tailor posting.txt --company "Desjardins" --lang fr
```

1. Evalúa el posting (fit + requirements). Si hay mismatch duro, se detiene sin generar nada.
2. Detecta Quebec por ubicación, empresa o idioma del posting → CV en francés, título "Développeur" (nunca "Ingénieur").
3. Elige el CV base de `data/cv/` por idioma y variante Kafka (por nombre de archivo: `fr`/`en`, `kafka`/`sin_kafka`).
4. Claude reescribe headline, resumen, bullets y skills **solo con lo que dicen tus documentos**; lo que falta va a "gaps", no al CV.
5. Guardrails deterministas: corrige el título en francés y marca cualquier tecnología del borrador que no aparezca en tus fuentes.
6. Escribe en `output/pending/<fecha>_<empresa>_<puesto>/`: `cv.md` (para el empleador) y `review.md` (evidencia por bullet, gaps, flags). Estado `pending_approval`: nada se envía.

## Respuestas a formularios (Fase 6)

Requiere la sección `[situation]` en `data/profile.toml` (cómo terminó tu último empleo, salario, disponibilidad...). Usa como ejemplo de tono tus respuestas previas en `data/qa_history/`.

```powershell
python -m job_assistant answer examples/forms/formulaire_quebec.txt --company "Desjardins" --role "Développeur Java"
```

- Detecta premisas falsas (p. ej. "suite à votre démission" cuando fue un layoff, o "your current employer" sin empleo actual) con reglas fijas **y** con Claude; las marca en vez de responder tal cual.
- Si falta un dato (salario, años exactos), deja un placeholder y lo marca como "Needs your input". Consentimientos y declaraciones legales siempre se marcan como "Your decision".
- Idioma: francés si el formulario está en francés o el posting es de Quebec. Salida en `output/pending/.../form_answers.md`.

## CRM de reclutadores + Follow-ups (Fase 7)

Hojas `Reclutadores` y `Seguimientos` del mismo Excel.

```powershell
python -m job_assistant contacts add --name "Ana Gómez" --agency TEKsystems --client Citibank --role "Senior Java Dev"
python -m job_assistant contacts touch "Ana Gómez" --status "Esperando respuesta" --next "Esperar feedback"
python -m job_assistant contacts list                 # STALE = más de FOLLOWUP_STALE_DAYS sin contacto
python -m job_assistant followup due                  # reclutadores y aplicaciones que necesitan seguimiento
python -m job_assistant followup draft --contact "Ana"   # o --company "Shopify" (aplicación del tracker)
python -m job_assistant followup sent --to "Ana Gómez"   # TÚ lo enviaste: se registra y actualiza el contacto
python -m job_assistant followup discard --to "Ana Gómez"
```

Memoria: no se genera un segundo mensaje al mismo reclutador o empresa si hay un borrador pendiente o uno enviado hace menos de `FOLLOWUP_COOLDOWN_DAYS` días (y en ese caso ni siquiera se llama a Claude).

## Discovery (Fase 8)

Nunca toca LinkedIn ni Indeed. Dos fuentes:

1. **Bandeja manual** `data/inbox/`: guarda ahí el texto de vacantes que veas (opcionalmente con líneas `Company:`, `Title:`, `Location:`, `URL:` arriba). Se procesan y se mueven a `data/inbox/processed/`.
2. **Páginas de empleo de empresas** vía las APIs públicas de solo lectura de Greenhouse, Lever y Ashby (una consulta por empresa). Lista en `data/discovery.toml` (ver `examples/discovery.example.toml`).

```powershell
python -m job_assistant discover                          # reporte en output/discovery/<fecha>/report.md
python -m job_assistant discover --check --top 5 --add    # + requirements check y alta en el tracker de las que pasan
python -m job_assistant daily                              # corrida diaria: busca nuevas -> las agrega a Postular -> genera CVs -> reporte
python -m job_assistant daily --no-cv                      # solo busca y llena la hoja (sin gastar IA)
```

Filtra por título, ubicación y que mencione Java/Spring/Kotlin; descarta lo que ya está en el tracker; ordena por fit. Cada vacante nueva se guarda como .txt listo para `tailor`.

## CVs a medida desde el Excel (flujo principal)

1. En la hoja **Postular** de `data/tracker.xlsx` agrega una fila por vacante: mínimo el **Link** (o pega la descripción si el link es de LinkedIn/Indeed). Opcional: Empresa, Puesto, Idioma (`auto`/`en`/`fr`), Notas para el CV.
2. Ejecuta:

```powershell
python -m job_assistant postular            # procesa filas con Estado vacío / Nuevo / Regenerar / Forzar
python -m job_assistant postular --row 5    # solo esa fila, sin importar su estado
```

Para cada fila: lee la descripción (una vez), revisa requisitos (si hay un obligatorio que no cumples, se detiene salvo `Forzar`), el **Agente Redactor** escribe el CV con tus hechos fijos (`data/cv_master.toml`) y el **Agente HR** lo revisa (veracidad, cobertura, tono humano, claridad, ATS) hasta 2 rondas. Se conserva la mejor ronda.

### Masters por rol (opcional, recomendado)

`data/roles.toml` define los roles a los que apuntas (Business Systems Analyst, Guidewire BA, Java Backend, Tech Lead...). Cada uno lleva `title_keywords` (para reconocer el título de la vacante), 10 `ats_keywords` y `verify_terms` (productos que el resume no puede afirmar si no están en tus hechos).

```powershell
python -m job_assistant masters                  # escribe un master por rol en output/master_resumes/
python -m job_assistant masters --role java-backend
```

Genera `Master_<Rol>_Resume_EN.pdf` (+ su `.json`), `roles_index.json` (rol → archivo y palabras clave de título) y `ROLES_REPORT.md` (qué palabras clave tienes y cuáles son brechas). Con ese índice, `postular` elige el master del rol más cercano al título de cada vacante, parte de él y lo adapta a la descripción; la columna **Master usado** del Excel dice cuál eligió y por qué. Sin `roles.toml`, `postular` funciona como antes. Los resumes nunca afirman un `verify_terms` que no esté en `cv_master.toml` (es un bloqueo en código), no usan rayas como puntuación y piden la fórmula XYZ solo donde tus hechos dan una medida.

**Costo:** cada corrida muestra tokens y costo estimado por vacante y el total. Para gastar menos: la 2.ª ronda solo ocurre si la 1.ª no alcanza promedio 7.0 con veracidad 7+ y sin bloqueos; la revisión de requisitos y el fiscalizador usan razonamiento bajo (`REVIEW_EFFORT` en `.env`: low/medium/high; vacío = default del modelo); el redactor usa el default. **Proveedores:** por defecto todo corre con Claude. DeepSeek es opcional (más barato): pon `DEEPSEEK_API_KEY` en `.env` y elige por rol `WRITER_PROVIDER` / `REVIEW_PROVIDER` (`anthropic` o `deepseek`), o por corrida con `--writer-provider` / `--review-provider`. `python -m job_assistant check --provider deepseek` prueba la conexión. Ojo: lo que se envía a DeepSeek sale de los servidores de Anthropic (perfil y datos de tu CV).

`ANTHROPIC_ENABLED=false` en `.env` bloquea cualquier llamada a la API de Claude (error claro en vez de gasto). `python -m job_assistant audit [--row N]` vuelve a revisar CVs ya generados con el revisor configurado (puntajes calibrados a la escala de Claude; un revisor DeepSeek no es más estricto que el de `postular`, solo da una segunda lectura). Protecciones sin costo de API: bloqueo de dominios/habilidades que no están en tus datos (banca, "secured", "regulated"...), detección de verbos de liderazgo inflados, y los requisitos de ubicación/modalidad nunca detienen un CV.

Estado `CV listo – revisar` = cumple esa regla; `CV con observaciones` = revisa las observaciones antes de enviar.

Salida: una carpeta por día y por corrida, `output/Resumenes 10-6/` (la segunda corrida del mismo día: `Resumenes 10-6 2`, luego `... 3`). Dentro, una subcarpeta por vacante (`<Empresa>_<Puesto>/`) con el PDF (`Felipe_Rodriguez_Resume_...` o `..._CV_...` en francés), `job_description.txt`, `revision_hr.md` y los JSON. La carpeta del día solo aparece cuando se genera el primer CV (una corrida sin resultados no deja carpetas vacías). Los CVs generados antes de este cambio siguen en `output/cv/`. La fila del Excel enlaza al PDF.

Idioma: el del posting (o la columna Idioma). En Quebec el titular nunca dice Engineer/Ingénieur.

## Otra persona en el mismo computador (perfiles)

No se duplica el proyecto: hay **un solo código y una carpeta de datos por persona**. Tu perfil es el de siempre (`data/`, `output/`); cada persona adicional vive en `profiles/<nombre>/` (ignorado por git) con su propio CV, Excel, índice y resultados.

```powershell
jobs.cmd novia setup        # crea sus carpetas y archivos de partida (seguro repetirlo: no pisa lo editado)
jobs.cmd novia index        # indexa sus CVs
jobs.cmd novia postular     # genera sus CVs desde su hoja Postular
```

`jobs.cmd <perfil> <comando>` es lo mismo que `python -m job_assistant --profile <perfil> <comando>`. Sin `--profile` se trabaja con tus datos; con él, la terminal escribe `[profile: nombre]` para que se vea de quién son los datos.

- **Aislamiento:** un perfil nunca hereda las rutas del `.env` compartido (`DATA_DIR`, `TRACKER_PATH`...): solo las suyas. Las keys y los modelos sí se comparten; ella puede usar su propia key en `profiles/<nombre>/.env`.
- **Datos de ejemplo:** el sistema se niega a generar CVs mientras `cv_master.toml` o `profile.toml` conserven el ejemplo (ALEX SAMPLE / Your Name).
- **Qué completa ella:** `profile.toml` (incluye `flexible_location = true/false`: si acepta mudarse y cualquier modalidad) y `cv_master.toml` (empleos, fechas, educación; en `[meta] field` su profesión, que se usa en los prompts).
- **Privacidad:** con DeepSeek, sus datos salen de los servidores de Anthropic; debe estar de acuerdo.

La primera vez descarga el modelo de embeddings local (~80 MB). El índice queda en `vectorstore/` (gitignored).

## Estructura

```
job_assistant/
├── config.py      # Settings desde .env (la key nunca se imprime)
├── llm.py         # Fábrica única del modelo Claude (LangChain)
├── __main__.py    # CLI: python -m job_assistant <comando>
└── agents/        # Se llena fase a fase
data/              # Tus datos reales (gitignored) — ver data/README.md
output/            # Borradores pendientes de aprobación (gitignored)
tests/
```

## Roadmap

1. ✅ Setup, estructura, .env, conexión a Claude
2. ✅ Profile / RAG Agent (pendiente: indexar tus documentos reales)
3. ✅ Scoring de fit + Requirements Check (pendiente: completar data/profile.toml)
4. ✅ Tracker Agent (Excel) (pendiente: probar con copia de tu tracker real)
5. ✅ Tailoring Agent + lógica Quebec (pendiente: probar con tus CVs reales)
6. ✅ Application Q&A Agent (pendiente: completar [situation] y data/qa_history/)
7. ✅ Recruiter CRM + Follow-up Agent (pendiente: cargar tus reclutadores reales)
8. ✅ Discovery Agent (pendiente: tu lista de empresas en data/discovery.toml)
9. Interview Prep Agent
10. Weekly Summary + integración con aprobación humana
