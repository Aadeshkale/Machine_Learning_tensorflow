'''
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=6tki-Aro_Uax
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# selecting dataset model from the tf.keras.dataset
mnist = tf.keras.datasets.fashion_mnist

# loading data in the from of labels
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()



# exploring sample images dataset using numpy,mathplotlib
np.set_printoptions(linewidth=200)
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])


# converting data into values into 0 to 1 using float division bez values in float improves ML performance

training_images = training_images/255.0
test_images = test_images/255.0

# creating model
'''
Sequential: That defines a SEQUENCE of layers in the neural network

Flatten: Remember earlier where our images were a square, when you printed them out? Flatten just takes that square and turns it into a 1 dimensional set.

Dense: Adds a layer of neurons Each layer of neurons need an activation function to tell them what to do. There's lots of options, but just use these for now.

Relu effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.

Softmax takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!

'''

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# compiling model

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
# training of model
model.fit(training_images, training_labels, epochs=5)

# predicting result for test data
model.evaluate(test_images, test_labels)


classifications = model.predict(test_images)

print(classifications[0])

print(test_labels[0])


