'''
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=DoWp43WxJDNT
'''

import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


train_horse_dir = 'horse-or-human/horses'
train_human_dir = 'horse-or-human/humans'

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
print(len(train_human_names))
print(len(train_horse_names))



# fallback class to stop at perticular accuracy
class Callback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

myCallback = Callback()



# building model for classification

model = tf.keras.Sequential([

    # Convolution of images for only effective features
    # first convolution
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # flattring the results
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# printing summery of model
model.summary()

# compiling model
# selecting appropriate model compiler and optimizer

from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])


# traning of model with tensorflow image generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory('horse-or-human/',
                                                    target_size=(300,300),
                                                    batch_size=128,
               # Since we use binary_crossentropy loss, we need binary labels
                                                    class_mode='binary',

                                                    )


model.fit(train_generator,steps_per_epoch=8,  epochs=15,verbose=1,callbacks=[myCallback])


# testing the model
import numpy as np
from tensorflow.keras.preprocessing import image

path = 'horse-test.jpg'
img = image.load_img(path,target_size=(300,300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

classes = model.predict(x, batch_size=10)
print(classes[0])

if classes[0]>0.5:
    print("{} is a human".format(path))
else:
    print("{} is a horse".format(path))
























