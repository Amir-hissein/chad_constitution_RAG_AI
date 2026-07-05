# Plan complet — RAG Constitution du Tchad (de zéro)

Un assistant qui répond aux questions des Tchadiens sur leur Constitution.
Stack : Python + Antigravity + LangChain + ChromaDB + BGE-M3 + LLM.

**Principe d'apprentissage :** tu écris le code toi-même, l'agent Antigravity
t'assiste quand tu bloques. C'est ce qui te fera vraiment comprendre le RAG.

---

## PHASE 0 — Installer les outils

### Python (obligatoire)
- Va sur python.org/downloads, prends **Python 3.12**.
- À l'installation, coche **« Add Python to PATH »** (case en bas, cruciale).
- Vérifie dans un terminal : `python --version` → doit afficher `Python 3.12.x`.

### Antigravity (ton éditeur, remplace VS Code)
- Télécharge sur antigravity.google, lance l'installeur.
- Premier lancement : choisis « Start fresh », ton thème, et le mode
  **« Agent-assisted »** (tu gardes le contrôle, l'IA aide).
- Connecte-toi avec ton compte Google → débloque Gemini gratuit.

### Ollama (optionnel — LLM 100% local et gratuit)
- Sur ollama.com. Puis dans un terminal : `ollama pull llama3.1`
- Pas indispensable : Antigravity te donne déjà Gemini gratuitement.

> Rappel : Python, Antigravity, Ollama = LOGICIELS (on les télécharge).
> LangChain, ChromaDB, PyMuPDF... = LIBRAIRIES (installées via pip, utilisées
> avec `import`, aucun site à visiter).

---

## PHASE 1 — Créer et ouvrir le projet

Dans le terminal intégré d'Antigravity (menu Terminal) :

```bash
mkdir rag-constitution
cd rag-constitution
```

Puis dans Antigravity : **Fichier → Ouvrir le dossier** → sélectionne `rag-constitution`.

---

## PHASE 2 — Environnement virtuel + librairies

L'environnement virtuel isole les librairies de ce projet.

```bash
python -m venv venv
```

Active-le selon ton système :

```bash
# Windows PowerShell
venv\Scripts\Activate.ps1
# Windows cmd
venv\Scripts\activate.bat
# Mac / Linux
source venv/bin/activate
```

Quand `(venv)` apparaît en début de ligne, installe tout d'un coup :

```bash
pip install pymupdf langchain langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-ollama langchain-openai sentence-transformers chromadb langgraph fastapi uvicorn python-dotenv
```

Dis à Antigravity d'utiliser ce venv :
`Ctrl+Shift+P` → « Python: Select Interpreter » → choisis celui avec `venv`.

Fige tes librairies : `pip freeze > requirements.txt`

---

## PHASE 3 — Structure du projet

```
rag-constitution/
├── venv/                      (déjà créé)
├── data/
│   └── constitution_tchad.pdf (mets ton PDF ici)
├── src/
│   ├── etape1_charger.py
│   ├── etape2_decouper.py
│   ├── etape3_4_indexer.py
│   ├── etape5_6_rag.py
│   └── etape7_graphe.py
├── chroma_db/                 (créé auto à l'étape 4)
├── .env
└── requirements.txt
```

---

## PHASE 4 — Coder les 7 étapes

Rythme : écris une étape → lance `python src/fichier.py` → observe → ajuste.
On ne passe à la suite que quand ça tourne.

### Étape 1 — Charger le PDF (fichier : etape1_charger.py)

```python
import fitz  # PyMuPDF
import re

def charger_pdf(chemin):
    doc = fitz.open(chemin)
    texte = ""
    for page in doc:
        texte += page.get_text()
    doc.close()
    return texte

def nettoyer(texte):
    texte = re.sub(r'-\n', '', texte)                                   # mots coupés
    texte = re.sub(r'(?<=[a-zéèêàû\.])\d{1,3}(?=\s*\n)', '', texte)      # n° de page collés
    texte = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', texte)                   # n° de page seuls
    texte = re.sub(r'[ \t]+', ' ', texte)                              # espaces multiples
    return texte

if __name__ == "__main__":
    texte = nettoyer(charger_pdf("data/constitution_tchad.pdf"))
    print(len(texte), "caractères")
    print(texte[:800])
    print("Article 92 trouvé à l'index :", texte.find("Article 92"))
```

**À vérifier :** texte propre, « Article 92 » trouvé, les chiffres réels
(âges, n° d'articles) préservés.

### Étape 2 — Découper par article (fichier : etape2_decouper.py)

Le cœur du projet : on découpe sur chaque article + on garde le n° en métadonnée.

```python
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from etape1_charger import charger_pdf, nettoyer

def detecter_titre(num):
    if num <= 12:   return "Titre I : De l'État et de la souveraineté"
    if num <= 64:   return "Titre II : Libertés, droits fondamentaux et devoirs"
    if num <= 109:  return "Titre III : Du pouvoir exécutif"
    if num <= 131:  return "Titre IV : Du pouvoir législatif"
    if num <= 154:  return "Titre V : Rapports exécutif / législatif"
    if num <= 172:  return "Titre VI : Du pouvoir judiciaire"
    return "Autre"  # complète avec le reste des Titres

def decouper_par_article(texte):
    pattern = r'(Article\s+(\d+)(?:er)?\s*:)'
    morceaux = re.split(pattern, texte)
    documents = []
    for i in range(1, len(morceaux), 3):
        marqueur = morceaux[i]
        numero = int(morceaux[i + 1])
        contenu = morceaux[i + 2].strip()
        documents.append(Document(
            page_content=f"{marqueur} {contenu}",
            metadata={"article": numero, "titre": detecter_titre(numero),
                      "source": "Constitution du Tchad 2025"}
        ))
    return documents

def decouper(texte):
    documents = decouper_par_article(texte)
    sous_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150, separators=["\n", ". ", " "])
    chunks = []
    for doc in documents:
        if len(doc.page_content) > 1500:
            chunks.extend(sous_splitter.split_documents([doc]))
        else:
            chunks.append(doc)
    return chunks

if __name__ == "__main__":
    texte = nettoyer(charger_pdf("data/constitution_tchad.pdf"))
    chunks = decouper(texte)
    print(f"{len(chunks)} chunks finaux")
    print(chunks[91].page_content[:200])
    print(chunks[91].metadata)
```

### Étapes 3 & 4 — Embeddings + indexation (fichier : etape3_4_indexer.py)

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from etape1_charger import charger_pdf, nettoyer
from etape2_decouper import decouper

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},           # "cuda" si GPU
        encode_kwargs={"normalize_embeddings": True})

def indexer():
    texte = nettoyer(charger_pdf("data/constitution_tchad.pdf"))
    chunks = decouper(texte)
    vs = Chroma.from_documents(
        documents=chunks, embedding=get_embeddings(),
        collection_name="constitution_tchad", persist_directory="./chroma_db")
    print("Indexé :", vs._collection.count(), "chunks")
    return vs

if __name__ == "__main__":
    indexer()  # 1er lancement : télécharge BGE-M3 (~2 Go), sois patient
```

> Règle d'or : le MÊME modèle d'embedding pour indexer ET pour chercher.

### Étapes 5 & 6 — Retrieval + génération (fichier : etape5_6_rag.py)

```python
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama   # ou: from langchain_openai import ChatOpenAI
from etape3_4_indexer import get_embeddings

vectorstore = Chroma(
    collection_name="constitution_tchad",
    embedding_function=get_embeddings(),
    persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model="llama3.1", temperature=0)
# Variante OpenAI : llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
"""Tu es un assistant juridique spécialisé dans la Constitution du Tchad.
Réponds à la question UNIQUEMENT à partir du contexte ci-dessous.
Si la réponse ne s'y trouve pas, dis "Cette information ne figure pas dans la Constitution."
Cite toujours le ou les numéros d'article sur lesquels tu t'appuies.

Contexte :
{context}

Question : {question}

Réponse (en français, claire et précise) :""")

def formater_contexte(docs):
    return "\n\n".join(f"[Article {d.metadata['article']}] {d.page_content}" for d in docs)

chaine = ({"context": retriever | formater_contexte,
           "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

if __name__ == "__main__":
    print(chaine.invoke("Quelles sont les conditions pour être candidat à la présidence ?"))
```

**Questions de test :** « à quel âge peut-on voter ? » (art. 6),
« durée du mandat du président ? » (art. 67), « l'esclavage est-il interdit ? » (art. 20),
et une piège hors-sujet : « capitale de la France ? » → doit refuser.

**À ce stade, ton RAG fonctionne.**

### Étape 7 — Orchestration LangGraph (fichier : etape7_graphe.py)

```python
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from etape5_6_rag import retriever, prompt, llm, formater_contexte

class EtatRAG(TypedDict):
    question: str
    documents: List[Document]
    reponse: str

def noeud_retrieve(etat):
    return {"documents": retriever.invoke(etat["question"])}

def noeud_generate(etat):
    contexte = formater_contexte(etat["documents"])
    rep = (prompt | llm | StrOutputParser()).invoke(
        {"context": contexte, "question": etat["question"]})
    return {"reponse": rep}

g = StateGraph(EtatRAG)
g.add_node("retrieve", noeud_retrieve)
g.add_node("generate", noeud_generate)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)
app = g.compile()

if __name__ == "__main__":
    r = app.invoke({"question": "Combien de temps dure le mandat d'un sénateur ?"})
    print(r["reponse"])
```

**Évaluation :** construis ~20 questions dont tu connais la bonne réponse ET
le bon article. Relance-les à chaque changement (chunking, k, prompt) pour
mesurer si tu améliores ou dégrades. Sans ça, tu bricoles à l'aveugle.

---

## PHASE 5 — Transformer en API FastAPI

Une fois le RAG maîtrisé, on l'emballe en API pour une future app mobile/web
(comme pour ebillet). Un endpoint `/question` qui prend une question et renvoie
la réponse + les articles cités. Lancement : `uvicorn main:app --reload`.
On construira cette phase ensemble quand le RAG de base tournera.

---

## Comment utiliser l'agent Antigravity intelligemment

- Fichier qui plante → surligne l'erreur, demande « explique cette erreur et
  comment la corriger » (pas « corrige tout »).
- Ligne pas comprise → surligne-la, demande « explique cette ligne ».
- Tu gardes la main, l'agent est ton tuteur. Une fois que tu maîtrises, tu
  pourras le lâcher en autonomie sur les parties répétitives.

## Ordre de travail conseillé

1. Phases 0 à 3 (install + structure) → confirme que `(venv)` s'affiche.
2. Étape 1 → observe le texte nettoyé, ajuste le nettoyage à ton PDF réel.
3. Une étape à la fois, en testant, jusqu'à l'étape 7.
4. Puis Phase 5 (API).
