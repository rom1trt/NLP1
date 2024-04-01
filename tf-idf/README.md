# Text Classification Comparison

This project compares the performance of TF-IDF (Term Frequency-Inverse Document Frequency) and other ùethods. We use a dataset of reviews with corresponding ratings to evaluate the effectiveness of these methods.

## Dataset

The dataset used for this project is `Reviews.csv`, which contains textual reviews and corresponding rating scores.

## Implementation

### Data Preparation

* We start by loading the dataset using `pd.read_csv` 
* In the second method, the text data is preprocessed by removing punctuation, converting text to lowercase, and tokenizing the text into words.

### TF-IDF Approach

* We use the `TfidfVectorizer` from `sklearn.feature_extraction.text` to convert the text data into TF-IDF feature vectors.
* The TF-IDF model is trained on the training set and evaluated on the testing set using classification metrics such as precision, recall, and F1-score.

## Results
comparison of TF-IDF results and other methods

### Naive Bayes

- **TF-IDF Results:**

**Precision:** Precision values range between 0.20 and 0.82.
**Recall:** Recall varies between 0.00 and 1.00 for different classes.
**F1 Score:** F1 scores range from 0.00 to 0.79.
**Accuracy:** The overall accuracy of both models is 0.65.

- **CounterVectorizer Results:**

**Precision:** Precision values range from 0.40 to 0.83.
**Recall:** Recall ranges between 0.24 and 0.87 for different classes.
**F1 Score:** F1 scores range from 0.31 to 0.85.
**Accuracy:** The overall accuracy of the model is 0.71.

### Analysis:

Compared to the provided results, your results seem to be less favorable in terms of precision, recall, and F1 score for most classes. However, the overall accuracy of your models is slightly lower than that of the provided results.

### Conclusion:

Although your results are not as good as those provided in terms of precision, recall, and F1 score, they are still relatively close. It may be necessary to explore other approaches or adjust model parameters to improve performance. Additionally, a deeper analysis of specific misclassification errors can help identify areas where the models need improvement.