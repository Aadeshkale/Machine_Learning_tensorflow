"""
https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras


# Define model

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


'''
The LOSS function measures the guessed answers against the known correct answers and measures how well or how badly it did.
It then uses the OPTIMIZER function to make another guess. Based on how the loss function went, it will try to minimize the loss. 
'''

model.compile(optimizer='sgd', loss='mean_squared_error')

# data for model to predict the next no in sequence

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# traning the model

model.fit(xs,ys,epochs=500)

# predicting new result using the model

print(model.predict([10.0]))






