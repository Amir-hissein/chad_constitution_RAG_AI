from typing import TypedDict, List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser       
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END
from rag_chain import retriever, prompt, llm, formater_contexte

# 1. L'ÉTAT : le dictionnaire qui circule dans le graphe
class EtatRAG(TypedDict):
    question: str
    documents: List[Document]
    reponse: str

# 2. LES NŒUDS : chacun remplit une partie de l'état
def noeud_retrieve(etat):
    return {"documents": retriever.invoke(etat["question"])}

def noeud_generate(etat):
    contexte = formater_contexte(etat["documents"])
    rep = (prompt | llm | StrOutputParser()).invoke(
        {"context": contexte, "question": etat["question"]})
    return {"reponse": rep}

# 3. LE GRAPHE : on relie les nœuds
g = StateGraph(EtatRAG)
g.add_node("retrieve", noeud_retrieve)
g.add_node("generate", noeud_generate)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)

# 4. COMPILATION : le graphe devient une app exécutable
app = g.compile()

if __name__ == "__main__":
    question = "Combien de temps dure le mandat d'un sénateur ?"
    resultat = app.invoke({"question": question})
    print("QUESTION :", question, "\n")
    print("RÉPONSE :", resultat["reponse"]) 