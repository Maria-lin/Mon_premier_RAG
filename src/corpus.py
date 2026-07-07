"""Lecture du corpus depuis le fichier CSV.

Chaque ligne du CSV est un chunk : id, text, source, categorie.
"""

import csv

from .config import CORPUS_CSV


def load_corpus(csv_path: str = CORPUS_CSV) -> list[dict]:
    """Lit le CSV et retourne une liste de chunks.

    Chaque chunk est un dictionnaire :
    {"id": ..., "text": ..., "metadata": {"source": ..., "categorie": ...}}
    """
    chunks = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": {
                        "source": row["source"],
                        "categorie": row["categorie"],
                    },
                }
            )
    return chunks


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"{len(corpus)} chunks charges")
    print("Premier chunk :", corpus[0])
