## Approche #1 
Un modèle RNN (Recurrent Neural Network) a été entraîné sur 10000 avis textuels provenant du dossier `data` en utilisant la librairie `keras`. 

Les principales étapes et détails du modèle sont :

- Prétraitement du texte : 
 - Mise en minuscules
 - Suppression de la ponctuation
 - Filtrage des mots vides

- Tokenization et encodage :
    - Tokenisation du texte en mots 
    - Encodage en séquences d'entiers via `Tokenizer` de Keras
    - Vocabulaire de taille 10000 (`max_features`)
    - Séquences paddées à une longueur de 100 (`max_length`)

- Architecture du modèle : 
    - Couche d'embedding : `Embedding(max_features, 32)`
    - 3 couches LSTM empilées : `LSTM(16, return_sequences=True)`, `LSTM(16, return_sequences=True)`, `LSTM(16)`  
    - Couche dense intermédiaire : `Dense(64, activation='relu')` 
    - Couche de dropout : `Dropout(0.5)`
    - Couche de sortie : `Dense(5, activation='softmax')`

- Entraînement : 
    - Optimiseur : Adam
    - Loss : categorical cross-entropy
    - Métrique : accuracy
    - Batch size : 128
    - Validation split : 0.2

Ce modèle RNN atteint une accuracy de 78.3% sur l'ensemble de test, permettant de prédire assez efficacement le score (entre 1 et 5) associé à un avis textuel. Les couches LSTM permettent de bien capturer les dépendances à long-terme dans les séquences de mots.

## Pistes d'amélioration

- Augmenter la taille du vocabulaire (`max_features`) 
- Faire varier la dimension d'embedding
- Essayer différentes tailles de LSTM
- Ajouter des couches LSTM supplémentaires
- Essayer des variantes de LSTM (BiLSTM, GRU, etc)
- Faire de l'ajustement d'hyperparamètres (learning rate, batch size, nombre d'epochs, etc)
- Utiliser des techniques de régularisation (L1/L2)
