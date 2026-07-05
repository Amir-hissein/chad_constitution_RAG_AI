import re
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_pdf import charger_pdf, nettoyer

# (article de départ, libellé du titre), dans l'ordre de la Constitution.
# Un article appartient au dernier titre dont le numéro de départ lui est <=.
TITRES = [
    (1,   "Titre I : De l'État et de la souveraineté"),
    (13,  "Titre II : Des libertés, des droits fondamentaux et des devoirs"),
    (65,  "Titre III : Du pouvoir exécutif"),
    (110, "Titre IV : Du pouvoir législatif"),
    (132, "Titre V : Des rapports entre le pouvoir exécutif et le pouvoir législatif"),
    (155, "Titre VI : Du pouvoir judiciaire"),
    (173, "Titre VII : Du Conseil constitutionnel"),
    (184, "Titre VIII : De la Cour des comptes"),
    (190, "Titre IX : De la Haute cour de justice"),
    (197, "Titre X : De la justice militaire"),
    (202, "Titre XI : Du Conseil économique, social, culturel et environnemental"),
    (207, "Titre XII : De la Commission nationale des droits de l'homme"),
    (214, "Titre XIII : De la Haute autorité des médias et de l'audiovisuel"),
    (220, "Titre XIV : Du Haut conseil des chefferies traditionnelles"),
    (225, "Titre XV : Des autorités traditionnelles et coutumières"),
    (230, "Titre XVI : De la Médiature de la République"),
    (235, "Titre XVII : De l'Agence nationale de gestion des élections"),
    (240, "Titre XVIII : De la défense nationale et de la sécurité"),
    (253, "Titre XIX : Des collectivités autonomes"),
    (272, "Titre XX : De la coopération, des traités et accords internationaux"),
    (277, "Titre XXI : De l'adoption d'un projet de Constitution"),
    (279, "Titre XXII : De la révision"),
    (283, "Titre XXIII : Des dispositions transitoires et finales"),
]

def detecter_titre(num):
    titre = TITRES[0][1]
    for debut, libelle in TITRES:
        if num >= debut:
            titre = libelle
        else:
            break
    return titre

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
    exemple = chunks[len(chunks) // 2]   # un chunk du milieu, quel que soit le nombre total
    print("Contenu :", exemple.page_content[:180])
    print("Métadonnées :", exemple.metadata)