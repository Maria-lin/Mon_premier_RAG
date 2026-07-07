"""Interface web du RAG : une page de test + une API JSON.

Lancement : uvicorn app:app --reload
Puis ouvrir http://127.0.0.1:8000
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.rag import RAG

app = FastAPI(title="Mon premier RAG")

# Initialise une seule fois au demarrage du serveur (base + moderateur + client)
rag = RAG()


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(q: Question):
    """Reçoit {"question": ...} et retourne la réponse, les sources et la modération."""
    return rag.answer_question(q.question)


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()
