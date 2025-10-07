# Finance Document Q&A API

This repository contains a FastAPI service for uploading finance documents (PDF, DOCX, TXT) and asking questions against them using Azure OpenAI.

Quick start

1. Create and activate a Python virtual environment.
2. Populate a `.env` file at the project root with DB and Azure credentials (see `.env.example` below).
3. Install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

4. Run locally:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Docker

Build and run the Docker image:

```powershell
docker build -t finance-qna .
docker run -p 8000:8000 --env-file .env finance-qna
```

Security

- Do NOT commit your `.env` file. It's included in `.gitignore`.
- Rotate your Azure key if it has been committed previously.
# backendfin
