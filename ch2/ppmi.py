import numpy as np
from util import preprocess, create_co_matrix, most_similar, ppmi


text="You say goodbye and I say hello."
corpus, word_to_id, id_to_word=preprocess(text)
vocab_size=len(word_to_id)
C=create_co_matrix(corpus, vocab_size)

#most_similar("you", word_to_id, id_to_word, C,top=5)
W=ppmi(C)

np.set_printoptions(precision=3)
print('C')
print(C)
print('-'*50)
print("PPMI")
print(W)

