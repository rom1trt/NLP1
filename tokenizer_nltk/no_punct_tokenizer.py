import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
import sklearn as sk
import sklearn.model_selection

data = pd.read_csv('../_data/Reviews.csv') # Loading the dataset
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(data['Text'], data['Score'], test_size=0.2) # Splitting in test and training sets

X_train = X_train[:len(X_train) // 5] # reducing the length of the training set to a fifth of the original size
X_test = X_test[:len(X_test) // 5] # reducing the length of the testing set to a fifth of the original size

text_strings_train = " ".join(X_train) # Put every rows in the training set in one string, separated by a whitespace
sents_train = sent_tokenize(text_strings_train) # Tokenize by sentences

text_strings_test = " ".join(X_test) # Put every rows in the testing set in one string, separated by a whitespace
sents_test = sent_tokenize(text_strings_test) # Tokenize by sentences

token_train = [nltk.RegexpTokenizer(r"\w+").tokenize(s) for s in sents_train] # Removes ALL punctuation and parenthesis etc...

token_test = [nltk.RegexpTokenizer(r"\w+").tokenize(s) for s in sents_test] # Removes ALL punctuation and parenthesis etc...

tokenized_train = [word for sublist in token_train for word in sublist]
tokenized_test = [word for sublist in token_test for word in sublist]

with open('data/training_data/no_punct_tokenizer_train.data', 'w') as f:
    f.writelines(word + '\n' for word in tokenized_train)

with open('data/testing_data/no_punct_tokenizer_test.data', 'w') as f:
    f.writelines(word + '\n' for word in tokenized_test)