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
6. Application Q&A Agent
7. Recruiter CRM + Follow-up Agent
8. Discovery Agent
9. Interview Prep Agent
10. Weekly Summary + integración con aprobación humana
