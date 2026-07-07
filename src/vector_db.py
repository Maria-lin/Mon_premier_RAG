"""Base vectorielle persistante : ChromaDB + sentence-transformers.

Comportement du constructeur :
- si une base existe deja sur disque -> on la recharge (aucun encodage) ;
- sinon, si des chunks sont fournis -> on la cree ;
- sinon -> erreur explicite.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from .config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL


class VectorDB:
    def __init__(self, chunks: list[dict] | None = None, path: str = CHROMA_PATH):
        self.client = chromadb.PersistentClient(path=path)

        existing = [c.name for c in self.client.list_collections()]

        if COLLECTION_NAME in existing:
            # La base existe deja : on la recharge et on lit le nom du modele
            # d'embedding dans les metadonnees de la collection. On charge CE
            # modele-la, pas celui de config.py : cela rend impossible le bug
            # silencieux ou l'on interroge une base indexee avec un modele A
            # en encodant les questions avec un modele B (similarites absurdes,
            # tres difficile a diagnostiquer).
            self.collection = self.client.get_collection(COLLECTION_NAME)
            model_name = self.collection.metadata["embedding_model"]
            self.model = SentenceTransformer(model_name)
            print(f"Base rechargee ({self.collection.count()} chunks, modele : {model_name})")
        elif chunks is not None:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "embedding_model": EMBEDDING_MODEL,
                    "hnsw:space": "cosine",
                },
            )
            self._index(chunks)
            print(f"Base creee ({self.collection.count()} chunks)")
        else:
            raise ValueError(
                f"Aucune base trouvee dans '{path}' et aucun chunk fourni : "
                "impossible de demarrer. Fournissez des chunks pour creer la base."
            )

    def _encode(self, texts: list[str]):
        # normalize_embeddings=True : vecteurs de norme 1, le produit scalaire
        # devient exactement la similarite cosinus vue en cours.
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _index(self, chunks: list[dict]):
        embeddings = self._encode([c["text"] for c in chunks])
        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[c["metadata"] for c in chunks],
        )

    def retrieve(self, question: str, n: int = 3) -> list[dict]:
        """Retourne les n chunks les plus proches de la question."""
        query_embedding = self._encode([question])
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]
