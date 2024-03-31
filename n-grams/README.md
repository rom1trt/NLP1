# Approches utilisées et Implementation

## Approche : Greedy #1
The Greedy approach selects the most probable token at each step without considering future consequences.

### Implementation
1. Calculate the n-gram probabilities.
2. Iterate through the tokens in the context.
3. Select the token with the highest probability based on the n-gram probabilities.
4. Repeat the process until reaching the maximum length or encountering a stopping condition.

## Approche : top k #2
The top k approach selects one of the k most probable tokens at each step, considering a broader range of possibilities compared to Greedy.

### Implementation
1. Calculate the n-gram probabilities.
2. Iterate through the tokens in the context.
3. Select the top k tokens based on their probabilities.
4. Choose one token from the top k tokens probabilistically, considering the weights of their probabilities.
5. Repeat the process until reaching the maximum length or encountering a stopping condition.

# Limitations
- For all n-gram, it is not very good at capturing long-range dependencies in the text data.
- The Greedy approach tends to loop, especially when encountering repetitive patterns in the data.

# Difficultés rencontrées
- Finding all stop cases for the algorithm.
- Taking the weights into consideration in the top k approach.
- Implementing Laplace smoothing to handle unseen n-grams effectively.

# Pistes d’améliorations
- Experimenting with different smoothing techniques to handle unseen n-grams more effectively.
- Exploring techniques to capture long-range dependencies in the text data more accurately.

# Results

## Greedy vs top k

# Results

## Greedy vs top k

The top k approach outperforms the Greedy approach in several aspects. Firstly, top k avoids the issue of looping, which is common in the Greedy approach, especially when encountering repetitive patterns in the data. Additionally, top k introduces variation into the generated sequences since it randomly selects from the top k most probable tokens, resulting in different outputs for the same input context. On the other hand, the Greedy approach tends to produce predictable outputs, often copying existing patterns in the data. 

However, it's worth noting that while top k offers advantages over Greedy, it may sometimes generate nonsensical sequences due to its probabilistic nature. 

## bi-gram, tri-gram, five-gram

Bi-gram: Bi-grams often produce nonsensical sequences because they only consider pairs of consecutive words, leading to a lack of context and coherence in the generated text.

Tri-gram: Tri-grams generally provide better results compared to bi-grams as they consider sequences of three consecutive words, allowing for some context to be captured. However, they may still lack sufficient connection between phrases or sentences.

Five-gram: Five-grams tend to produce more coherent and contextually relevant sequences compared to bi-grams and tri-grams. By considering sequences of five consecutive words, they capture more context and dependencies in the text. However, they may also tend to copy more directly from the training text, potentially leading to less diverse outputs.

## Perplexity

Bi-gram : 98 when tested with train tokens and 610 with test tokens
Tri-gram : 14 when tested with train tokens and 1838 with test tokens
Five-gram : 3 when tested with train tokens and 13933 with test tokens



