"""Interface web (Streamlit) pour le RAG sur la Constitution du Tchad.

Lancement :  PYTHONPATH=src streamlit run src/app.py
(ou simplement :  streamlit run src/app.py  — le dossier src est ajouté au path)

L'interface réutilise TA chaîne RAG existante (rag_chain.py) : rien n'est réécrit.
"""
import streamlit as st

st.set_page_config(page_title="Constitution du Tchad — Assistant", page_icon="🇹🇩")


@st.cache_resource(show_spinner="Chargement du modèle et de l'index (une seule fois)…")
def charger_rag():
    """Charge la chaîne RAG une seule fois, puis la garde en mémoire.

    L'import est fait ICI (pas en haut du fichier) pour que le chargement
    lourd de BGE-M3 et de ChromaDB ne se fasse qu'au premier appel, en cache.
    """
    from rag_chain import chaine, retriever, verifier_ollama
    return chaine, retriever, verifier_ollama


# --- En-tête ---
st.title("🇹🇩 Assistant — Constitution du Tchad")
st.caption("Pose une question ; l'assistant répond uniquement d'après le texte de la Constitution et cite les articles.")

chaine, retriever, verifier_ollama = charger_rag()

# --- Vérification d'Ollama ---
ok, msg = verifier_ollama()
if not ok:
    st.error(msg)
    st.stop()
st.success(msg)

# --- Questions d'exemple ---
EXEMPLES = [
    "Le Tchad est-il un État laïque ?",
    "Quelles sont les langues officielles du Tchad ?",
    "Quelles sont les conditions pour être candidat à la présidence ?",
    "Combien de temps dure le mandat d'un sénateur ?",
    "À qui appartient l'initiative de la révision de la Constitution ?",
]

with st.sidebar:
    st.header("Exemples de questions")
    for ex in EXEMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

# --- Champ de question ---
question = st.text_input(
    "Ta question :",
    key="question",
    placeholder="Ex. : Qui est le Chef de l'État ?",
)

if question:
    # 1) Articles trouvés par la recherche
    with st.spinner("Recherche des articles pertinents…"):
        docs = retriever.invoke(question)

    # 2) Réponse générée par le LLM
    with st.spinner("Rédaction de la réponse par llama3.1…"):
        reponse = chaine.invoke(question)

    st.subheader("Réponse")
    st.write(reponse)

    with st.expander(f"📚 Articles consultés ({len(docs)})"):
        for d in docs:
            st.markdown(
                f"**Article {d.metadata['article']}** "
                f"— *{d.metadata.get('titre', '')}*"
            )
            st.write(d.page_content)
            st.divider()
