# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser
# pyrefly: ignore [missing-import]
from langchain_core.runnables import RunnablePassthrough
# pyrefly: ignore [missing-import]
from langchain_ollama import ChatOllama
from build_index import get_embeddings

import json
import urllib.request
import urllib.error

MODELE = "llama3.1"
OLLAMA_URL = "http://localhost:11434"

def verifier_ollama(modele=MODELE):
    """Vérifie qu'Ollama tourne et que le modèle est disponible.
    Renvoie (ok: bool, message: str) pour un message clair à l'utilisateur."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            modeles = [m["name"] for m in json.load(r).get("models", [])]
    except (urllib.error.URLError, OSError):
        return False, (f"❌ Ollama ne répond pas sur {OLLAMA_URL}.\n"
                       f"   Démarre-le avec :  ollama serve")
    if not any(nom.split(":")[0] == modele for nom in modeles):
        return False, (f"❌ Le modèle '{modele}' n'est pas installé.\n"
                       f"   Installe-le avec :  ollama pull {modele}")
    return True, f"✅ Ollama OK — modèle '{modele}' disponible."

vectorstore = Chroma(
    collection_name="constitution_tchad",
    embedding_function=get_embeddings(),
    persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

llm = ChatOllama(model=MODELE, temperature=0)   # ← ton modèle ; temperature=0 = pas de fantaisie

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
    import sys
    ok, msg = verifier_ollama()
    print(msg)
    if not ok:
        sys.exit(1)

    question = "Quelles sont les conditions pour être candidat à la présidence ?"
    print("\nQUESTION :", question, "\n")

    # DEBUG : que trouve la recherche ?
    docs = retriever.invoke(question)
    print("=== ARTICLES TROUVÉS PAR LA RECHERCHE ===")
    for d in docs:
        print("  → Article", d.metadata["article"], ":", d.page_content[:80].replace("\n", " "))
    print()

    print("RÉPONSE :", chaine.invoke(question))