# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
# pyrefly: ignore [missing-import]
import chromadb
from load_pdf import charger_pdf, nettoyer
from split_documents import decouper

COLLECTION = "constitution_tchad"
CHROMA_DIR = "./chroma_db"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},           # "cuda" si tu avais un GPU
        encode_kwargs={"normalize_embeddings": True})

def indexer():
    texte = nettoyer(charger_pdf("data/constitution_tchad.pdf"))
    chunks = decouper(texte)

    # Idempotence : on repart d'une base propre pour éviter les doublons
    # si build_index.py est relancé plusieurs fois.
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
        print(f"🗑️  Ancienne collection '{COLLECTION}' supprimée.")
    except Exception:
        pass  # la collection n'existait pas encore : c'est normal au 1er lancement

    vs = Chroma.from_documents(
        documents=chunks, embedding=get_embeddings(),
        collection_name=COLLECTION, persist_directory=CHROMA_DIR)
    print("=== TEST DES ÉTAPES 3 & 4 ===\n")
    nb = vs._collection.count()
    if nb == len(chunks):
        print(f"✅ RÉUSSI : {nb} chunks indexés dans ChromaDB (= {len(chunks)} chunks créés)")
    else:
        print(f"⚠️  {nb} indexés pour {len(chunks)} chunks — relance après avoir supprimé chroma_db/")
    return vs

if __name__ == "__main__":
    indexer()  # 1er lancement : télécharge BGE-M3 (~2 Go) puis calcule les vecteurs — sois patient