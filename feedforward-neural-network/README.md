# Approches utilisées et Implementation

## Approche #1

Réseau de Neurones Feedforward

Prétraitement des Données

    Jeu de Données : Avis textuels avec scores associés.
    Mise en Minuscules : Conversion de tout le texte en minuscules.
    Suppression de la Ponctuation.
    Filtrage des Mots Vides : Utilisation de NLTK.
    Division des Données : 80 % pour l'entraînement, 20 % pour les tests.

Extraction des Caractéristiques

    Vectorisation : Vectoriseur TF-IDF de scikit-learn.
    Nombre Maximum de Caractéristiques : 1000.
    Seuils de Fréquence Documentaire Minimum et Maximum : Fixés respectivement à 7 et 0.8.
    Filtrage des Mots Vides en Anglais.

Modèle de Réseau de Neurones

    Architecture :
        Couche d'Entrée : Taille égale au nombre de caractéristiques de TF-IDF.
        Couches Denses : Trois couches avec respectivement 512, 256 et 128 neurones, toutes utilisant l'activation ReLU.
        Couche de Sortie : Couche softmax avec 5 neurones (un pour chaque catégorie de score).
    Compilation :
        Fonction de Perte : Entropie Croisée Catégorielle.
        Optimiseur : Adam.
        Métrique : Précision.

Entraînement

    Époques : 10.
    Taille de Lot : 32.

Évaluation et Résultats

    Évaluation du Modèle : Utilisation de la précision sur l'ensemble de test.
    Affichage de la Perte et de la Précision après évaluation.

## Approche #2

TODO

# Limitations

Les réseaux de neurones feedforward n'intègrent pas bien le contexte des mots dans une phrase

# Difficultés rencontrées

TODO

# Pistes d’améliorations

Expérimentation avec différents hyperparamètres, tels que le nombre de neurones dans chaque couche, la taille du lot et le taux d'apprentissage.
Essayer différentes architectures, ajouter plus de couches ou utiliser différentes fonctions d'activation.
Utiliser des techniques de régularisation (L1, L2) pour prévenir le surajustement.
