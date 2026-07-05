import re
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_pdf import charger_pdf, nettoyer

def detecter_titre(num):
    if num <= 12:   return "Titre I : De l'État et de la souveraineté"
    if num <= 64:   return "Titre II : Libertés, droits fondamentaux et devoirs"
    if num <= 109:  return "Titre III : Du pouvoir exécutif"
    if num <= 131:  return "Titre IV : Du pouvoir législatif"
    if num <= 154:  return "Titre V : Rapports exécutif / législatif"
    if num <= 172:  return "Titre VI : Du pouvoir judiciaire"
    return "Autre"  # articles 173-285 : à compléter plus tard

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

    print("=== TEST DE L'ÉTAPE 2 ===\n")

    if len(chunks) > 100:
        print(f"✅ TEST 1 RÉUSSI  : {len(chunks)} chunks créés (> 100)")
    else:
        print(f"❌ TEST 1 ÉCHOUÉ  : seulement {len(chunks)} chunks")

    # Chaque chunk a-t-il bien un numéro d'article en métadonnée ?
    if all("article" in c.metadata for c in chunks):
        print("✅ TEST 2 RÉUSSI  : tous les chunks ont une métadonnée 'article'")
    else:
        print("❌ TEST 2 ÉCHOUÉ  : certains chunks n'ont pas de numéro d'article")

    print("\n=== Exemple : un chunk et ses métadonnées ===")
    print("Contenu :", chunks[91].page_content[:180])
    print("Métadonnées :", chunks[91].metadata) 