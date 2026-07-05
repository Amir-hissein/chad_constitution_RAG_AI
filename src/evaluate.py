"""Évaluation automatique du RAG sur un jeu de questions à réponse connue.

On mesure DEUX choses séparément :
  1. Recall du retrieval : le bon article est-il dans les documents récupérés ?
     (objectif et rapide — ne dépend pas du LLM)
  2. Qualité de la réponse : contient-elle les mots-clés attendus,
     ou le bon refus pour une question hors-sujet ?

Usage :  PYTHONPATH=src python src/evaluate.py
"""
import json
import sys
from pathlib import Path

from rag_chain import retriever, chaine, verifier_ollama

FICHIER_QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.json"
PHRASE_REFUS = "ne figure pas"   # cf. le prompt dans rag_chain.py


def articles_recuperes(question):
    """Numéros d'article renvoyés par le retriever pour une question."""
    return {d.metadata["article"] for d in retriever.invoke(question)}


def reponse_ok(cas, reponse):
    """La réponse générée est-elle correcte pour ce cas ?"""
    texte = reponse.lower()
    if cas.get("refus_attendu"):
        return PHRASE_REFUS in texte
    return all(mot.lower() in texte for mot in cas.get("mots_cles", []))


def evaluer():
    cas = json.loads(FICHIER_QUESTIONS.read_text(encoding="utf-8"))

    retrieval_ok = 0
    reponses_ok = 0
    n_retrieval = 0   # on ne compte le retrieval que pour les questions ayant un article attendu

    print(f"\n=== ÉVALUATION SUR {len(cas)} QUESTIONS ===\n")
    for c in cas:
        attendus = set(c.get("articles_attendus", []))
        trouves = articles_recuperes(c["question"])

        # 1. Retrieval (ignoré pour les questions hors-sujet sans article attendu)
        if attendus:
            n_retrieval += 1
            hit = bool(attendus & trouves)
            retrieval_ok += hit
            marque_r = "✅" if hit else "❌"
            detail_r = f"attendu {sorted(attendus)}, trouvé {sorted(trouves)}"
        else:
            marque_r = "➖"
            detail_r = "(hors-sujet : pas d'article attendu)"

        # 2. Réponse générée
        reponse = chaine.invoke(c["question"])
        rep_ok = reponse_ok(c, reponse)
        reponses_ok += rep_ok
        marque_a = "✅" if rep_ok else "❌"

        print(f"Q: {c['question']}")
        print(f"   Retrieval {marque_r}  {detail_r}")
        print(f"   Réponse   {marque_a}  {reponse.strip()[:120]}...")
        print()

    print("=== SCORES ===")
    print(f"Retrieval (bon article récupéré) : {retrieval_ok}/{n_retrieval} "
          f"= {100*retrieval_ok/n_retrieval:.0f}%")
    print(f"Réponses correctes               : {reponses_ok}/{len(cas)} "
          f"= {100*reponses_ok/len(cas):.0f}%")


if __name__ == "__main__":
    ok, msg = verifier_ollama()
    print(msg)
    if not ok:
        sys.exit(1)
    evaluer()
