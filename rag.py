"""Orchestrateur RAG : moderation, retrieval, puis appel au LLM principal."""

import anthropic
from dotenv import load_dotenv

from config import LLM_MODEL, RAG_PROMPT_FILE, TOP_K
from moderator import Moderator
from vector_db import VectorDB


class RAG:
    def __init__(self):
        load_dotenv()
        self.client = anthropic.Anthropic()
        self.moderator = Moderator()
        self.db = VectorDB()

    def answer_question(self, question: str) -> str:
        # La moderation passe avant tout : si c'est une injection, on ne
        # contacte jamais le LLM principal. L'ordre est une decision de securite.
        decision = self.moderator.moderate(question)
        if decision["is_prompt_injection"]:
            return "Je ne peux pas repondre a cette question."

        chunks = self.db.retrieve(question, TOP_K)

        with open(RAG_PROMPT_FILE, encoding="utf-8") as f:
            prompt_template = f.read()

        chunks_text = "\n".join(f"- {c['text']}" for c in chunks)
        system_prompt = prompt_template.replace("{{Chunks}}", chunks_text)

        response = self.client.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text
