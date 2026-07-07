"""Test du retrieval : 5 questions dont on connait le bon chunk.

A lancer AVANT de brancher le LLM : si le bon chunk ne remonte pas en tete,
le probleme est ici et pas ailleurs.
"""

from corpus import load_corpus
from vector_db import VectorDB

QUESTIONS = [
    "Quelle est la couleur du chat de Bob ?",
    "Comment s'appelle le chien d'Alice ?",
    "Ou dort le perroquet de Diego ?",
    "Qui est le maire de Villebrume ?",
    "Combien d'habitants compte Villebrume-les-Cuilleres ?",
]

if __name__ == "__main__":
    # Cree la base au premier lancement, la recharge ensuite
    db = VectorDB(chunks=load_corpus())

    for question in QUESTIONS:
        print(f"\n=== {question}")
        for chunk in db.retrieve(question, n=3):
            print(f"  [{chunk['id']}] (dist={chunk['distance']:.3f}) {chunk['text']}")
