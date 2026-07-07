# Mon premier RAG — Mini-TP M2 MD5

Un RAG minimal mais complet : ChromaDB + sentence-transformers + API Anthropic + agent modérateur + interface web.

Le corpus (`05_corpus_rag.csv`, 200 chunks) est constitué de faits inventés (un chat bleu nommé Henri, un village appelé Villebrume-les-Cuillères...) : ces faits n'existent nulle part sur Internet, donc si le système répond juste, c'est forcément grâce au retrieval.

## Architecture

Trois briques, chacune dans son fichier :

| Brique | Fichiers | Rôle |
|---|---|---|
| Base vectorielle | `src/vector_db.py` (+ `src/corpus.py`) | Crée ou recharge une base ChromaDB persistée, encode les chunks avec `distiluse-base-multilingual-cased-v2` (normalisé → similarité cosinus), retrouve les k chunks les plus proches d'une question |
| Agent modérateur | `src/moderator.py` + `prompts/moderator.txt` | Détecte les tentatives de prompt injection, sortie JSON forcée par schéma : `{"is_prompt_injection": true/false}` |
| RAG orchestrateur | `src/rag.py` + `prompts/rag_system.txt` | Pipeline complet : modération → retrieval (top-3) → prompt système à trous (`{{Chunks}}`) → appel au LLM |

Le cœur du RAG vit dans le package `src/` ; `app.py` et les scripts `test_*.py` restent à la racine, ce sont des points d'entrée qui l'utilisent. S'y ajoutent `src/config.py` (tous les noms de modèles et chemins à un seul endroit) et les prompts en fichiers texte (un prompt se retravaille sans toucher au code).

**Ordre du pipeline = décision de sécurité** : si le modérateur détecte une injection, le LLM principal n'est jamais contacté.

**Détail important** : le nom du modèle d'embedding est stocké dans les métadonnées de la collection ChromaDB et relu au rechargement — impossible d'interroger la base avec un autre modèle que celui qui l'a indexée.

## Installation (pas à pas, même sans expérience Git/Python)

**Prérequis** : avoir [Python](https://www.python.org/downloads/) (3.10 ou plus récent) et [Git](https://git-scm.com/downloads) installés. Pour vérifier, ouvrir un terminal et taper `python --version` puis `git --version` : si une version s'affiche, c'est bon.

**1. Récupérer le projet**

```bash
git clone https://github.com/Maria-lin/Mon_premier_RAG.git
cd Mon_premier_RAG
```

**2. Créer un environnement virtuel** (un espace Python isolé pour ce projet, pour ne pas mélanger ses dépendances avec le reste de la machine)

```bash
python -m venv venv
```

**3. Activer l'environnement virtuel** (à refaire à chaque nouvelle session de terminal)

```bash
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
```

Si ça a fonctionné, le terminal affiche `(venv)` au tout début de la ligne.

**4. Installer les dépendances du projet**

```bash
pip install -r requirements.txt
```

Ça installe ChromaDB, sentence-transformers, l'API Anthropic, FastAPI, etc. Peut prendre quelques minutes la première fois.

**5. Créer le fichier `.env` avec sa clé API**

Ce projet a besoin d'une clé API Anthropic pour appeler Claude. Copier le fichier modèle fourni :

```bash
copy .env.example .env          # Windows
cp .env.example .env            # macOS / Linux
```

Puis ouvrir `.env` et remplacer `sk-ant-votre-cle-ici` par une vraie clé, obtenue gratuitement sur https://console.anthropic.com (rubrique "API Keys").

`.env` est listé dans `.gitignore` : il ne sera jamais envoyé sur GitHub, chacun garde sa propre clé en local.

**6. Vérifier que tout fonctionne**

```bash
python test_retrieval.py
```

Le tout premier lancement télécharge le modèle d'embedding (~500 Mo, une seule fois) et construit la base vectorielle dans `chroma_db/` à partir de `data/05_corpus_rag.csv`. Les lancements suivants rechargent cette base instantanément. Si les 5 questions affichent la bonne réponse en tête de liste, l'installation est réussie.

## Utilisation

```bash
# 1. Tester le retrieval (crée la base au premier lancement, la recharge ensuite)
python test_retrieval.py

# 2. Tester le pipeline complet (les 4 tests de la mise à l'épreuve)
python test_pipeline.py

# 3. Lancer l'interface web
uvicorn app:app --reload
# puis ouvrir http://127.0.0.1:8000
```

L'interface affiche la réponse (rendu Markdown), les sources utilisées avec leurs distances, et signale les questions bloquées par le modérateur.

## Évaluation

Une grille de 20 questions (`questions_test.txt`) couvre 5 niveaux : factuel simple, chiffres précis, synthèse multi-chunks, contradictions, refus/sécurité. **Score obtenu : 19/20** — le point perdu vient des questions dont l'information est éclatée sur plus de 3 chunks (limite structurelle du top-3 imposé par le sujet ; dans ce cas le système répond avec ce qu'il a et signale ce qui manque, sans inventer).

Résultats de la mise à l'épreuve (section 6 du sujet) :
- injection → bloquée par le modérateur, 0 chunk transmis, LLM principal jamais appelé ;
- sans modérateur → le prompt système résiste mais la question est traitée : la défense par prompt seule n'est pas une garantie ;
- hors corpus (« capitale du Japon ») → « je ne sais pas », aucune invention ;
- affirmation fausse (« le chat de Bob est vert ») → contradiction signalée avec la version du corpus.

## Réponses aux questions du TP

**1. Qui intercepte la question piégée, et à quel moment exact du pipeline ?**

L'agent modérateur, et c'est le **tout premier maillon** : la question lui est soumise avant la recherche vectorielle et avant tout appel au LLM principal. S'il détecte une injection, `answer_question` retourne immédiatement un refus — le retrieval n'a pas lieu et le LLM principal n'est jamais contacté.

**2. Que se passerait-il sans agent modérateur ?**

Testé (test 2 de `test_pipeline.py`) : la question piégée atteint le LLM principal. Dans notre essai le prompt système a résisté, mais cette résistance n'est **pas garantie** — elle dépend du modèle et de la formulation de l'attaque. Sans modérateur, la seule ligne de défense est le prompt lui-même ; avec modérateur, l'attaque est bloquée avant d'exposer le LLM principal (défense en profondeur).

**3. Pourquoi confier la modération à un appel dédié plutôt qu'une consigne dans le prompt du RAG ?**

Séparation des responsabilités (un appel = une mission, sortie JSON binaire testable), et ordre des opérations : une consigne dans le prompt du RAG agit *pendant* l'appel au LLM principal — l'attaque y est déjà exposée ; le modérateur agit *avant*. De plus, l'injection s'adresse précisément au prompt du RAG : c'est fragile de lui demander de se défendre lui-même.

**4. Quel bug la métadonnée `embedding_model` de la collection rend-elle impossible ?**

Le bug silencieux où la base a été indexée avec un modèle A mais les questions sont encodées avec un modèle B (config modifiée entre-temps) : vecteurs incompatibles, distances absurdes, retrieval aléatoire, **aucun message d'erreur**. En relisant le nom du modèle depuis la collection au rechargement, la cohérence indexation/interrogation est garantie par construction.

**5. Pourquoi normaliser les embeddings ?**

Vecteurs de norme 1 → le produit scalaire devient exactement la similarité cosinus : on compare les textes par leur direction (leur sens), pas par la longueur de leurs vecteurs.

**6. Les consignes du prompt système et le problème que chacune prévient**

| Consigne | Problème évité |
|---|---|
| Tous les chunks ne sont pas forcément utiles | Le retrieval retourne toujours k chunks, même peu pertinents : le modèle ne doit pas se croire obligé de tous les utiliser |
| Triés du plus au moins pertinent | Aide à arbitrer quand deux chunks se recouvrent ou se contredisent |
| Répondre uniquement depuis la base | L'hallucination : mélanger la mémoire du modèle avec le corpus |
| Hors périmètre → dire qu'on ne sait pas | Inventer une réponse plausible sur un sujet absent du corpus |
| Contradiction → signaler avec la bonne version | Valider par complaisance une affirmation fausse de l'utilisateur |

## Bonus : seuil de distance (branche `main_v2`)

Si le meilleur chunk est trop éloigné de la question (`distance > DISTANCE_THRESHOLD`), la réponse est précédée d'un avertissement de fiabilité. **Calibration mesurée** sur la grille de 20 questions : meilleures distances en-corpus entre 0,33 et 0,66 ; hors-corpus ≥ 0,75 → seuil fixé à **0,70**, entre les deux populations.

## Structure du projet

```
├── .env                  (local, jamais committé — voir .env.example)
├── .env.example          modèle du fichier .env à copier
├── .gitignore
├── requirements.txt
├── src/                  cœur du RAG (package Python)
│   ├── config.py         constantes : modèles, chemins, top-k, seuil
│   ├── corpus.py         lecture du CSV
│   ├── vector_db.py      brique 1 : base vectorielle persistante
│   ├── moderator.py      brique 2 : agent modérateur
│   └── rag.py            brique 3 : orchestrateur
├── app.py                interface web FastAPI (point d'entrée)
├── static/index.html     page de test
├── test_retrieval.py     validation du retrieval (5 questions)
├── test_pipeline.py      mise à l'épreuve (4 tests de la section 6)
├── questions_test.txt    grille d'évaluation (20 questions)
├── prompts/              prompts système (hors du code)
├── data/
│   └── 05_corpus_rag.csv le corpus (200 chunks)
└── chroma_db/            base vectorielle (générée, jamais committée)
```

## Note sur l'adaptation du sujet

Le sujet original utilisait Groq (`llama-3.3-70b-versatile` + un modèle « safeguard » pour la modération). Ce projet utilise l'API Anthropic : même architecture, mêmes responsabilités. Le modérateur n'est pas un modèle spécialisé mais un appel dédié avec prompt strict et sortie JSON forcée par schéma — la décision de sécurité (filtrer avant d'appeler le LLM principal) reste identique.
