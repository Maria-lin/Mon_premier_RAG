"""Agent moderateur : detecte les tentatives de prompt injection avant tout appel au RAG."""

import json

import anthropic

from config import MODERATOR_MODEL, MODERATOR_PROMPT_FILE


class Moderator:
    def __init__(self):
        self.client = anthropic.Anthropic()
        with open(MODERATOR_PROMPT_FILE, encoding="utf-8") as f:
            self.system_prompt = f.read()

    def moderate(self, question: str) -> dict:
        """Retourne {"is_prompt_injection": bool} pour la question donnee."""
        response = self.client.messages.create(
            model=MODERATOR_MODEL,
            max_tokens=256,
            system=self.system_prompt,
            messages=[{"role": "user", "content": question}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"is_prompt_injection": {"type": "boolean"}},
                        "required": ["is_prompt_injection"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        return json.loads(response.content[0].text)
