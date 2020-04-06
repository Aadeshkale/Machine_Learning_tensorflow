import tensorflow as tf
model = tf.keras.models.load_model('mymodel')

# testing model
import numpy as np
from tensorflow.keras.preprocessing import image

path = '../horse-test.jpg'
img = image.load_img(path,target_size=(300,300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

classes = model.predict(x, batch_size=10)
print(classes[0])

if classes[0]>0.5:
    print("{} is a human".format(path))
else:
    print("{} is a horse".format(path))

