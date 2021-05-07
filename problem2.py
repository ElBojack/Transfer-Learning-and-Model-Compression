import numpy as np
import tensorflow as tf
import os
import matplotlib.pylab as plt
import time
converter = tf.lite.TFLiteConverter.from_saved_model('saved_models/model_1/') # path to the SavedModel directory
tflite_model = converter.convert()

os.makedirs(os.path.join(os.getcwd(),'saved_tflite_models'), exist_ok = True)

with open('./saved_tflite_models/model_1.tflite', 'wb') as f:
  f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='saved_tflite_models/model_1.tflite')

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.resize_tensor_input(input_details[0]['index'], (32, 224, 224, 3))
interpreter.resize_tensor_input(output_details[0]['index'], (32, 224, 224, 3))

interpreter.allocate_tensors()

data_root = 'archive/Train'
IMAGE_SHAPE = (224, 224)
datagen_args = dict(rescale=1./255, validation_split=.20)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_args)
valid_generator = valid_datagen.flow_from_directory(str(data_root), subset="validation", shuffle=True, target_size=IMAGE_SHAPE)

image_batch, label_batch = next(iter(valid_generator))
model_interpreter_start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], image_batch)
interpreter.invoke()
used_time = time.time() - model_interpreter_start_time
result = interpreter.get_tensor(output_details[0]['index'])

def getAccuracy(labels, results):
    size = len(labels)
    ll = list()

    for i in range(size):
        ll.append(np.where(labels[i] == 1)[0][0] == results[i])

    return "Accuracy: {0}%.".format(ll.count(True) * 100 / size)

def isClass(label, res):
    return np.where(label == 1)[0][0] == res

result = result.argmax(axis=-1)

print('used_time:{}'.format(used_time))
plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(image_batch[n])
    correct = isClass(label_batch[n], result[n])
    color = "green" if correct else "red"
    plt.title(result[n], color=color)
    plt.axis('off')
    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
plt.show()

print(getAccuracy(label_batch, result))
'''(3） The compressed model's size is 8.64M and the pre-compressed model in problem 1 is 10.8M'''
'''(4) It use 1.9sec on average(result after multiple runs) '''
'''(5) With tf lite model, we achieve an accuracy of 97% on average (result atfer multiple runs).
This tf lite model was adapted from a problem 1 model with 20 epoch and 98% accuracy at the end of the last epoch.
Compared to the problem 1 model (20 epoch), we lost 1% accuracy on average. It is negligible. We can conclude that the compression doesn't impact the accuracy.
'''