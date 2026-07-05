# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
from load_pdf import charger_pdf, nettoyer
from split_documents import decouper

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},           # "cuda" si tu avais un GPU
        encode_kwargs={"normalize_embeddings": True})

def indexer():
    texte = nettoyer(charger_pdf("data/constitution_tchad.pdf"))
    chunks = decouper(texte)
    vs = Chroma.from_documents(
        documents=chunks, embedding=get_embeddings(),
        collection_name="constitution_tchad", persist_directory="./chroma_db")
    print("=== TEST DES ÉTAPES 3 & 4 ===\n")
    nb = vs._collection.count()
    if nb == len(chunks):
        print(f"✅ RÉUSSI : {nb} chunks indexés dans ChromaDB (= {len(chunks)} chunks créés)")
    else:
        print(f"⚠️  {nb} indexés pour {len(chunks)} chunks — relance après avoir supprimé chroma_db/")
    return vs

if __name__ == "__main__":
    indexer()  # 1er lancement : télécharge BGE-M3 (~2 Go) puis calcule les vecteurs — sois patient