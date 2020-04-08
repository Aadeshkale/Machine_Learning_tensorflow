'''
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb#scrollTo=BZSlp3DAjdYf
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# building model
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
print(model.summary())

# compile model with appropriate loss and optimizer

model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=1e-4),metrics=['acc'])

# preparing input image dataset using ImageDataGenerator for auto label and other features

# image data generator class with image argumentation parameters

train_data_gen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest'
)

validation_data_gen = ImageDataGenerator(rescale=1/255)

# selecting sample data set directory

train_data = train_data_gen.flow_from_directory('dataset/train/',
                               target_size=(150,150),
                               batch_size=32,
                               class_mode='binary')

validation__data = validation_data_gen.flow_from_directory('dataset/validation/',
                               target_size=(150,150),
                               batch_size=32,
                               class_mode='binary')

# model training
history = model.fit_generator(train_data,
                    validation_data=validation__data,
                    steps_per_epoch=8,
                    epochs=20,
                    verbose=1,
                    validation_steps=8,
)

# checking for accuracy graph

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


