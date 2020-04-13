import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


sentences = [
    "I love to code",
    "I love my dog",
    "I love my cat",
    "I love my family and and all my friends,relatives"
]

tokenizer = Tokenizer(num_words=100,oov_token="<OOV>")
# creating word index from sentences
tokenizer.fit_on_texts(sentences)
print("Words Index:\n",tokenizer.index_word)
# creating sequences form sentences
sequences = tokenizer.texts_to_sequences(sentences)
print("Sequences:\n",tokenizer.texts_to_sequences(sentences))
# creating all sentences with same length that is padding
pad = pad_sequences(sequences=sequences)
print("Padding of sequences:\n",pad)


