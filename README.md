# Projet : Recherche d’Images par le Contenu (CBIR) et Détection de Near-Duplicates avec SimCLR

## Description du Projet
Ce projet a pour objectif de développer un modèle auto-supervisé basé sur SimCLR permettant l'apprentissage de représentations visuelles adaptées aux tâches de recherche d’images par le contenu (CBIR) et plus spécifiquement à la détection de duplicatas proches (« near duplicates »). Le projet a été évalué sur un dataset médical, en l'occurrence des radiographies thoraciques (COVID-19 Chest X-ray Dataset).

### Objectifs principaux
1. Adapter le modèle SimCLR à la spécificité du domaine médical.
2. Entraîner et valider le modèle sur un dataset de radiographies.
3. Évaluer la qualité des représentations visuelles avec des métriques quantitatives comme la mAP et la précision@5.
4. Implémenter une solution pour la recherche d’images similaires et la détection de duplicatas proches.

---

## Datasets

### Dataset Utilisé
- **COVID-19 Chest X-ray Dataset** : Un dataset de radiographies thoraciques contenant plusieurs classes (ex. Pneumonie virale, COVID-19, Tuberculose, etc.).

### Préparation des Données
- **Transformations** : Normalisation, redimensionnement à `224x224`, et data augmentation (flips horizontaux, recadrage aléatoire, etc.).
- **Split des données** :
  - 80 % pour l’entraînement.
  - 10 % pour la validation.
  - 10 % pour le test.

---

## Modèle : SimCLR

### Architecture
- **Base Encoder** : ResNet pré-entrané.
- **Projection Head** : Deux couches linéaires avec activation ReLU et normalisation batch.

### Perte : NT-Xent Loss
- Basée sur la similarité cosinus.
- Encourage les vues augmentées d'une même image à être proches dans l'espace des embeddings.

---

## Entraînement

### Hyperparamètres
- Optimiseur : Adam
- Taux d’apprentissage : 0.001
- Température pour NT-Xent Loss : 0.5
- Nombre d’époques : 50
- Batch size : 64

### Protocole d’évaluation
1. **Validation croisee** : Suivi de la perte NT-Xent sur l'ensemble de validation.
2. **Visualisation des embeddings** : Utilisation de t-SNE pour réduire les dimensions des embeddings et observer leur séparation.

---

## Résultats

### Métriques d’évaluation
- **Mean Precision@5** : 33.14 %
- **Mean Average Precision (mAP)** : 92.72 %

### Observations
- Les embeddings générés permettent une séparation efficace des classes dans l'espace visuel.
- Une meilleure précision est obtenue sur des classes ayant des exemples visuellement distincts.

### Visualisation
1. **t-SNE** : Montre une séparation claire entre différentes classes de radiographies.
2. **Content-Based Image Retrieval (CBIR)** :
   - Recherche d’images similaires à une requête basée sur la distance cosinus.
   - Les duplicatas proches sont correctement identifiés dans la plupart des cas.

---

## Code Source

### Structure du Projet
1. **`Project.py`** : Contient le code source du projet.
2. **`simclr_covid_model.pth_epoch50.pth`** : Sauvegarde du model entrainé après 50 epochs.
3. **`test_embeddings.npy`** : Embeddings générés du test set.
4. **`test_labels.npy`** : Labels du test set.

---

## Conclusion
Ce projet démontre comment l’apprentissage auto-supervisé avec SimCLR peut être adapté au domaine médical pour des tâches comme le CBIR et la détection de duplicatas proches. Les résultats obtenus mettent en avant l’efficacité de la méthodologie employée et ouvrent la voie à des améliorations futures pour des domaines similaires.

