"""Constantes du projet : noms des modèles et chemins, définis à un seul endroit."""

# Modèle d'embedding (sentence-transformers)
EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"

# Modèles Anthropic
LLM_MODEL = "claude-opus-4-8"
MODERATOR_MODEL = "claude-opus-4-8"

# Base vectorielle
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "corpus_rag"

# Corpus
CORPUS_CSV = "data/05_corpus_rag.csv"

# Nombre de chunks récupérés par question
TOP_K = 3

# Prompts
RAG_PROMPT_FILE = "prompts/rag_system.txt"
MODERATOR_PROMPT_FILE = "prompts/moderator.txt"
