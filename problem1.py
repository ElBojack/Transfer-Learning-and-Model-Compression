from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import time

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

data_root = 'archive/Train'
arr = np.loadtxt("archive/Train.csv", delimiter=",", skiprows=1, dtype="str")
arr = np.array(arr[:, 6])
class_names = np.unique(arr)
print(class_names)
batch_size = 32
img_height = 224
img_width = 224
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(data_root),
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"  # @param {type:"string"}
IMAGE_SHAPE = (224, 224)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)
for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Labe batch shape: ", label_batch.shape)
    break

feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3))
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(image_data.num_classes, activation='softmax')
])

model.summary()
predictions = model(image_batch)
predictions.shape
print(predictions.shape)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['acc'])


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()


steps_per_epoch = np.ceil(image_data.samples / image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit(image_data, epochs=2,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[batch_stats_callback])
'''
Our Model can accurately predict which image belongs to which class. At the end of the second epoch the accuracy rating is 88%. 
(32, 43)
Epoch 1/2
1226/1226 [==============================] - 114s 89ms/step - loss: 0.9316 - acc: 0.7351
Epoch 2/2
1226/1226 [==============================] - 110s 90ms/step - loss: 0.4128 - acc: 0.8818
'''

plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0, 2])
plt.plot(batch_stats_callback.batch_losses)
plt.show()

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(batch_stats_callback.batch_acc)
plt.show()

predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]

label_id = np.argmax(label_batch, axis=-1)

plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(image_batch[n])
    color = "green" if predicted_id[n] == label_id[n] else "red"
    plt.title(predicted_label_batch[n].title(), color=color)
    plt.axis('off')
    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
plt.show()

t = time.time()

export_path = f"./saved_models/model_1"
model.save(export_path)

export_path

