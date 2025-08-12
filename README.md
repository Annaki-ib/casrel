# HBT - End-to-End Relation Extraction with BERT

Projet d'extraction de relations à partir de textes basé sur un modèle BERT end-to-end.

---

## Structure du projet

- **align_with_tokens.py** : fonctions pour aligner les entités avec les tokens produits par le tokenizer.
- **data_loader.py** : chargement et préparation des données, génération des batches pour l’entraînement.
- **model.py** : définition du modèle, fonctions d’entraînement et d’évaluation.
- **run.py** : script principal pour lancer l’entraînement ou l’évaluation.
- **utils.py** : fonctions utilitaires diverses (extraction d’entités, métriques, tokenizer).

---

## Installation & Prérequis

- Python 3.6+
- TensorFlow (compatible Keras)
- NumPy
- Autres librairies : argparse, etc.

Installe les dépendances avec (un `requirements.txt`) :


pip install -r requirements.txt
Préparation des données
Les datasets doivent être placés dans :


data/<dataset>/ ces fichiers sont meme dans le repo tplinker
avec les fichiers suivants :

train_triples.json (données d'entraînement)

dev_triples.json (données de validation)

test_triples.json (données de test)

rel2id.json (relations avec leur identifiant)

Utilisation
Entraînement

python run.py --train=True --dataset=NYT
Lance l'entraînement sur le dataset choisi (ici NYT_star).

Évaluation / Test

python run.py --train=False --dataset=NYT_star
Charge les poids sauvegardés et évalue sur le jeu de test.

Configuration
Modèle BERT pré-entraîné utilisé : cased_L-12_H-768_A-12.

Les poids et configurations BERT doivent être dans pretrained_bert_models/.

GPU configuré automatiquement dans run.py.

Résultats
Les résultats des tests sont sauvegardés dans :



results/<dataset>/test_result.json
