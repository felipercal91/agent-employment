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
3. Scoring de fit + Requirements Check
4. Tracker Agent (Excel)
5. Tailoring Agent + lógica Quebec
6. Application Q&A Agent
7. Recruiter CRM + Follow-up Agent
8. Discovery Agent
9. Interview Prep Agent
10. Weekly Summary + integración con aprobación humana
