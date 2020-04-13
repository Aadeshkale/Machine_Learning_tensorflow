'''
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb#scrollTo=it1c0jCiNCIM
'''
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import numpy as np
# selecting dataset ImageDataGenerator with image argumentaion parameters


# callback class
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.8):
            print("\nReached 80% accuracy so cancelling training!")
            self.model.stop_training = True
obj = myCallback()

train_data_gen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='nearest',
                                    )
validation_data_gen = ImageDataGenerator(rescale=1./255)

# selecting data directories with label structure

train_data = train_data_gen.flow_from_directory('dataset/rps/',target_size=(150,150))
validation_data = validation_data_gen.flow_from_directory('dataset/rps-test-set',target_size=(150,150))

# creating model for prediction
model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(64,(3,3),input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    # This is the second convolution
    tf.keras.layers.Conv2D(64,(3,3),input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results
    tf.keras.layers.Flatten(),
    # use of dropout to remove unneccessary neuron with same weight
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax'), # activation set to be softmax for multiclass classification
])
# summary of model
model.summary()


# model compilation

model.compile(optimizer=RMSprop(lr=0.0001)
              ,loss='categorical_crossentropy',
              metrics=['accuracy'])

# model Training

model.fit_generator(train_data,
                    epochs=25,
                    validation_data=validation_data,
                    verbose=1,
                    callbacks=[obj],
                    )

# model result test
path = 'test_img_1.jpg'
img = image.load_img(path,target_size=(150,150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

classes = model.predict(x, batch_size=10)
print('This {} is:'.format(path),classes)
