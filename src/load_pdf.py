# pyrefly: ignore [missing-import]
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
    texte = re.sub(r'[ \t]+', ' ', texte)                               # espaces multiples
    return texte

if __name__ == "__main__":
    texte = nettoyer(charger_pdf("data/constitution_tchad.pdf"))

    print("=== TEST DE L'ÉTAPE 1 ===\n")

    # Test 1 : le texte n'est pas vide
    if len(texte) > 10000:
        print(f"✅ TEST 1 RÉUSSI  : {len(texte)} caractères extraits (> 10000)")
    else:
        print(f"❌ TEST 1 ÉCHOUÉ  : seulement {len(texte)} caractères — PDF mal lu ?")

    # Test 2 : les articles sont repérables
    pos = texte.find("Article 92")
    if pos > 0:
        print(f"✅ TEST 2 RÉUSSI  : 'Article 92' trouvé à l'index {pos}")
    else:
        print(f"❌ TEST 2 ÉCHOUÉ  : 'Article 92' introuvable (index {pos})")

    # Test 3 : le texte est du français lisible
    if "Tchad" in texte and "suffrage" in texte:
        print("✅ TEST 3 RÉUSSI  : texte lisible ('Tchad' et 'suffrage' présents)")
    else:
        print("❌ TEST 3 ÉCHOUÉ  : mots attendus absents — texte corrompu ?")

    print("\n=== Aperçu du texte nettoyé (200 premiers caractères) ===")
    print(texte[:200])