# Logistic Regression on Product Reviews

This notebook demonstrates the implementation of logistic regression for sentiment analysis on a dataset of product reviews. The goal is to classify the reviews into different rating categories (1-5) based on the text content.

## Approach

1. **Data Loading**: The notebook starts by loading the product review dataset from a CSV file using pandas.

2. **Data Preprocessing**: The text data is vectorized using the CountVectorizer from scikit-learn. This converts the text into numerical feature vectors that can be used as input to the machine learning models.

3. **Data Splitting**: The dataset is split into training and testing sets using scikit-learn's `train_test_split` function.

4. **Model Training**: Two different approaches are used for training the logistic regression model:
   a. **Classic Approach**: The scikit-learn `LogisticRegression` model is trained directly on the vectorized data using the `fit` method.
   b. **Custom PyTorch Model**: A custom logistic regression model is implemented using PyTorch. The model is trained using the PyTorch data loader and optimizer.

5. **Model Evaluation**: After training, the models are evaluated on the test set. The following metrics are computed and reported:

- Accuracy
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix

## Requirements

The following libraries are required to run this notebook:

- pandas
- numpy
- torch
- scikit-learn

## Usage

1. Install the required libraries.
2. Run the notebook cells in order.
3. The results of the model evaluation will be printed in the output.
