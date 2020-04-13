import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
sentences = [
    "I love my work",
    "This is my home!",
]
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
words_dictionary = tokenizer.index_word
print(words_dictionary)
