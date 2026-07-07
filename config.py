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
CORPUS_CSV = "05_corpus_rag.csv"

# Nombre de chunks récupérés par question
TOP_K = 3

# Seuil de distance : au-dela, le meilleur chunk est juge trop eloigne de la
# question et la reponse est precedee d'un avertissement de fiabilite.
# Calibre sur la grille de 20 questions : en-corpus <= 0.66, hors-corpus >= 0.75.
DISTANCE_THRESHOLD = 0.70

# Prompts
RAG_PROMPT_FILE = "prompts/rag_system.txt"
MODERATOR_PROMPT_FILE = "prompts/moderator.txt"
