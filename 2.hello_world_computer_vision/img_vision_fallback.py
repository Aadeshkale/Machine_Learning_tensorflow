import tensorflow as tf

class Callback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.8):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True


myCallback = Callback()


mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
# training of model
model.fit(training_images, training_labels, epochs=10,callbacks=[myCallback])

# predicting result for test data
model.evaluate(test_images, test_labels)


classifications = model.predict(test_images)

print(classifications[0])

print(test_labels[0])

