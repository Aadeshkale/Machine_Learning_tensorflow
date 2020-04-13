import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
sentences = [
    "I love My Parents",
    "I love My GF",
    "I love My Friends",
    "I love My Dpg",
    "I love My Cats",
]
tokenizer = Tokenizer(num_words=100,oov_token="<OOV>") # here oov_token represents the word which is not on dictionary
tokenizer.fit_on_texts(sentences)
print("Words dictionary:",tokenizer.index_word)
# generating sequences
sequences = tokenizer.texts_to_sequences(sentences)
print("sentences generated for training data:",sequences)


sample_sentences = [
    "I really love My Parents",
    "I love My Job",
]
print("sentences generated for sample data",tokenizer.texts_to_sequences(sample_sentences))
