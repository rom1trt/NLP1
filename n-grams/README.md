## Results Analysis:
n is the n-gram
k is the applied smoothing
The first 100K texts of the dataset were used (we get around 7 millions tokens). 
# Normal tokenizer (split by space)
n=2 k=1
perplexity on train data : 4016
perplexity on test data : 5010

n=3 k=1
perplexity on train data : 25739
perplexity on test data : 44527

n=5 k=1
perplexity on train data : 57298
perplexity on test data : 140665

n=2 k=0.0001
perplexity on train data : 108
perplexity on test data : 512

n=3 k=0.0001
perplexity on train data : 29
perplexity on test data : 1811

n=5 k=0.0001
perplexity on train data :17
perplexity on test data : 23722

# Turn all to lowercase
n=2 k=1
perplexity on train data : 3120
perplexity on test data : 3835

n=3 k=1
perplexity on train data : 20888
perplexity on test data : 35511

n=5 k=1
perplexity on train data : 49353
perplexity on test data : 120580

n=2 k=0.0001
perplexity on train data : 111
perplexity on test data :461

n=3 k=0.0001
perplexity on train data : 28
perplexity on test data : 1500

n=5 k=0.0001
perplexity on train data : 15
perplexity on test data : 19827

# Turn all to lowercase and seperate punctuation
n=2 k=1
perplexity on train data : 465
perplexity on test data : 518

n=3 k=1
perplexity on train data : 2824
perplexity on test data : 4083

n=5 k=1
perplexity on train data : 10900
perplexity on test data : 24147

n=2 k=0.0001
perplexity on train data : 73
perplexity on test data : 141

n=3 k=0.0001
perplexity on train data : 19
perplexity on test data : 254

n=5 k=0.0001
perplexity on train data : 6
perplexity on test data : 3045



We have implemented 2 ways to do text generation using the n-grams, a greedy method taking the next most probable token and the to k method, choosing the next token among the top k token (the more probable they were to be the next token, the more probably they will be picked)
## Greedy vs top k

The top k approach outperforms the Greedy approach in several aspects. Firstly, top k avoids the issue of looping, which is common in the Greedy approach, especially when encountering repetitive patterns in the data. Additionally, top k introduces variation into the generated sequences since it randomly selects from the top k most probable tokens, resulting in different outputs for the same input context. On the other hand, the Greedy approach tends to produce predictable outputs, often copying existing patterns in the data. 

However, it's worth noting that while top k offers advantages over Greedy, it may sometimes generate nonsensical sequences due to its probabilistic nature. 

## bi-gram, tri-gram, five-gram

Bi-gram: Bi-grams often produce nonsensical sequences because they only consider pairs of consecutive words, leading to a lack of context and coherence in the generated text.

Tri-gram: Tri-grams generally provide better results compared to bi-grams as they consider sequences of three consecutive words, allowing for some context to be captured. However, they may still lack sufficient connection between phrases or sentences.

Five-gram: Five-grams tend to produce more coherent and contextually relevant sequences compared to bi-grams and tri-grams. By considering sequences of five consecutive words, they capture more context and dependencies in the text. However, they may also tend to copy more directly from the training text, potentially leading to less diverse outputs.

## Tokenizer impact

# tokenize_by_space
The basic tokenizer splits the text solely based on space, serving as the foundation for comparison with other tokenizers. 

# tokenize_by_space_and_lowercase
This tokenizer lowercases the tokens which reduces the vocabulary size. Lowercasing allows for better consistency in token representation and simplifies the tokenization process. It seems to enable the top k method to vary more, possibly due to the reduced vocabulary size and increased uniformity in token representation.


# tokenize_by_space_lowercase_and_punctuation
This tokenizer splits punctuation, resulting in less coherent outputs as punctuation seems to appear in unexpected places more frequently. However, this approach also introduces an advantage by fostering the generation of new and varied text. By treating punctuation marks as separate entities, the function creates tokens that are less specific and more versatile, encouraging the production of novel content. Despite heavy drawbacks in coherence, this method contributes to the diversification of generated text.

## k=1 (Laplace smoothing) VS k = 0.0001
We get better perplexity and more coherent text generation with k=0.0001 then with the laplace smoothing. The laplace smoothing seems to distort the context.
