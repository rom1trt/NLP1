# Tokenizer NLTK

## How to use it

The first thing you need is to install the libraries :
```pip install sklearn nltk pandas```  

You have to create a folder "data/" and then the folders "data/training_data/" and "data/testing_data/"  
After that you can chose one of the scripts to create a training set and a testing set. 
 
```python3 basic_tokenizer.py``` for exemple will tokenize the data_set with the basic method.
These scripts will create a .data file in the training and testing repositories, that you can use in your code by retrieving them with :
```python
  with open('path/to/datafile', 'r') as f:
    words = f.readlines()
  words = [word.strip() for word in words]
```
The variable "words" is now your dataset that you can use to train your model.


## How it works

The first thing we do is loading the dataset inside a DataFrame.  

Then we separate the dataset with 80% of the data for the training set and 20% for the testing set.

We saw that we have a lot of tokens, so just after, we only use a fifth of the total dataset. The final number of tokens goes from 40 000 000 for the training set to 8 000 000, which is still enough to properly train a model. We can use the full dataset later, when we have working models.  

After getting our training and testing sets, we use the ```join()``` method to Put all the reviews one after the other. This makes it possible to use the ```sent_tokenize()``` method of the nltk library to separate every sentence from each other.  

Finally, we can create the set of Tokens using the nltk library. For every sentence in the list we have, we will use a method to tokenize them. We use 3 methods here :
- Basic
- Whitespace
- Wordpunctuation
- Regex (Remove all punctuation)

Done, you have you list of Tokens now.
