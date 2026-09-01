# Apprentissage Non Supervisé - Segmentation Salariale (K-Means) & Réduction de Dimension (PCA)

## Description
Ce projet explore deux techniques clés du **Machine Learning non supervisé** avec **Scikit-learn** :
1. La **segmentation de salariés** (K-Means) à partir de leur âge et salaire.
2. La **réduction de dimension (PCA)** appliquée à la reconnaissance de chiffres manuscrits (dataset `digits`), combinée à une classification par **régression logistique**.

Le notebook illustre comment structurer un pipeline de clustering robuste (normalisation, choix du nombre de clusters) et comment la réduction de dimension peut simplifier un modèle de classification tout en conservant l'essentiel de l'information.

---

## Objectifs

**Partie 1 - Clustering (K-Means)**
* Explorer la relation entre âge et salaire (`salaires.csv`).
* Appliquer un premier K-Means (k=3) sur les données brutes.
* Construire une **Pipeline** (MinMaxScaler + KMeans) pour normaliser avant le clustering.
* Visualiser les clusters et leurs centroïdes (retransformés à l'échelle d'origine).
* Déterminer le nombre optimal de clusters avec la **méthode du coude (SSE)** et le **score de silhouette**.

**Partie 2 - Réduction de dimension (PCA) + Classification**
* Charger le jeu de données `digits` de Scikit-learn (images 8x8 de chiffres manuscrits).
* Entraîner une régression logistique de référence sur les données normalisées (64 dimensions).
* Appliquer une **PCA** pour réduire la dimensionnalité tout en conservant 95 % de la variance.
* Réentraîner le modèle sur les données réduites et comparer les performances.

---

## Technologies utilisées
* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Structure du projet
```
.
├── projet.ipynb
├── salaires.csv
└── README.md
```

---

## Résultats

**Clustering K-Means**
* Le score de silhouette est maximal pour **k = 3 clusters (≈ 0,356)**, confirmant que 3 groupes de salariés se distinguent naturellement selon l'âge et le salaire.
* La méthode du coude (SSE) corrobore ce choix, avec une baisse marquée de l'inertie jusqu'à k=3 puis une décroissance plus progressive.
* La normalisation des données (MinMaxScaler) via une Pipeline améliore la cohérence du clustering par rapport à un K-Means appliqué sur les données brutes.

**PCA + Régression Logistique (digits)**
* Le modèle de régression logistique de référence (64 variables) atteint une **précision de 96,9 %** sur l'ensemble de test.
* La PCA réduit les données de **64 à 40 dimensions** tout en conservant **95,1 % de la variance totale**.
* Réentraîné sur les données réduites, le modèle conserve d'excellentes performances (**précision d'entraînement ≈ 99,8 %**), démontrant qu'une grande partie de l'information utile est concentrée dans un nombre réduit de composantes principales.

---

## Lancement
1. Cloner le dépôt ou télécharger le notebook.
2. Installer les dépendances :
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
3. Ouvrir le notebook `projet.ipynb`.
4. Exécuter les cellules dans l'ordre.

---

## Compétences développées
* Clustering avec K-Means
* Normalisation des données (MinMaxScaler, StandardScaler) et Pipelines Scikit-learn
* Détermination du nombre optimal de clusters (méthode du coude, score de silhouette)
* Réduction de dimension avec l'Analyse en Composantes Principales (PCA)
* Classification multi-classe avec régression logistique
* Évaluation de modèles (classification report, matrice de confusion)
* Machine Learning non supervisé et supervisé avec Scikit-learn

---

## Auteur
Emmanuel YOHORE
