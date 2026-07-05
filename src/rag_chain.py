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

vectorstore = Chroma(
    collection_name="constitution_tchad",
    embedding_function=get_embeddings(),
    persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model="llama3.1", temperature=0)   # ← ton modèle ; temperature=0 = pas de fantaisie

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
    question = "Quelles sont les conditions pour être candidat à la présidence ?"
    print("QUESTION :", question, "\n")

    # DEBUG : que trouve la recherche ?
    docs = retriever.invoke(question)
    print("=== ARTICLES TROUVÉS PAR LA RECHERCHE ===")
    for d in docs:
        print("  → Article", d.metadata["article"], ":", d.page_content[:80].replace("\n", " "))
    print()

    print("RÉPONSE :", chaine.invoke(question))