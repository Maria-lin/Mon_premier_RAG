"""Orchestrateur du RAG : modération -> retrieval -> prompt à trous -> LLM.

Ordre des opérations = décision de sécurité : si le modérateur détecte une
injection, le LLM principal n'est JAMAIS contacté.
"""

import anthropic
from dotenv import load_dotenv

from config import LLM_MODEL, RAG_PROMPT_FILE, TOP_K
from corpus import load_corpus
from moderator import Moderator
from vector_db import VectorDB

REFUSAL_MESSAGE = (
    "Question refusée : l'agent modérateur a détecté une tentative de détournement."
)


class RAG:
    def __init__(self, use_moderator: bool = True):
        load_dotenv()
        self.client = anthropic.Anthropic()
        self.moderator = Moderator(self.client)
        self.use_moderator = use_moderator
        # Recharge la base si elle existe, la cree depuis le CSV sinon
        self.db = VectorDB(chunks=load_corpus())
        with open(RAG_PROMPT_FILE, encoding="utf-8") as f:
            self.prompt_template = f.read()

    def answer_question(self, question: str) -> dict:
        """Déroule le pipeline complet et retourne un dictionnaire :
        {"answer": ..., "chunks": [...], "moderation": {...}}
        """
        # 1. Moderation, AVANT tout appel au LLM principal
        if self.use_moderator:
            decision = self.moderator.moderate(question)
            if decision["is_prompt_injection"]:
                return {"answer": REFUSAL_MESSAGE, "chunks": [], "moderation": decision}
        else:
            decision = {"is_prompt_injection": None}  # moderateur desactive

        # 2. Recuperation des chunks les plus proches
        chunks = self.db.retrieve(question, n=TOP_K)

        # 3. Prompt systeme a trous : le marqueur {{Chunks}} est remplace
        chunks_text = "\n".join(
            f"{i + 1}. [{c['id']}] {c['text']}" for i, c in enumerate(chunks)
        )
        system_prompt = self.prompt_template.replace("{{Chunks}}", chunks_text)

        # 4. Appel au LLM : message system (prompt + chunks) et message user (question)
        response = self.client.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        answer = next(b.text for b in response.content if b.type == "text")
        return {"answer": answer, "chunks": chunks, "moderation": decision}


if __name__ == "__main__":
    rag = RAG()
    result = rag.answer_question("Quelle est la couleur du chat de Bob ?")
    print("Réponse :", result["answer"])
    print("Sources :", [c["id"] for c in result["chunks"]])
