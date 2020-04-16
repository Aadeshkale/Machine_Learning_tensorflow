import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku


# opening sample words file
file = open("dataset_wrords/shakespeare.txt",'r')

# reading the sentences from file
sentences = file.readlines()
print("sentences:",sentences)

# creating word index
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.index_word
print("word_index:",word_index)

# length of total words
len_words = len(word_index) + 1
print("len_words:",len_words)

# generating sequences from sentences
sequences = tokenizer.texts_to_sequences(sentences)
print("sequences:",sequences)

# finding maximum length sentence creating padding of sentences
sentence_max_len = max([len(i) for i in sentences])
pad = pad_sequences(sequences,maxlen=sentence_max_len,padding='pre')

# creating data and labels for model training
predictors, label = pad[:,:-1],pad[:,-1]


# used to convert array of labeled data(from 0 to nb_classes-1) to one-hot vector.
# label = ku.to_categorical(label, num_classes=len_words)

# defining model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len_words,100,input_length=sentence_max_len-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150,return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(len_words/2,activation='relu'),
    tf.keras.layers.Dense(len_words,activation='softmax'),
])

model.summary()


# model compilation
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# model training
model.fit(predictors, label, epochs=50, verbose=1)

# testing model
seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100

for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=sentence_max_len-1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=0)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)


















