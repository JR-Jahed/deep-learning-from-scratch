"""
Train the same models using tensorflow and check how they fare
"""


import keras
import numpy as np
import tensorflow as tf
import time

np.set_printoptions(suppress=True)

# Disable GPU usage globally
tf.config.set_visible_devices([], 'GPU')

width = 64
height = 64
channels = 3

total_images = 100
classes = 5
epochs = 30
batch_size = 32

images = np.random.randint(0, 256, (total_images, height, width, channels), dtype=np.uint8)
images = images / 255.0

labels = np.random.randint(0, classes, (total_images,))

initializer = keras.initializers.GlorotUniform()

model = keras.Sequential([
    keras.Input(shape=(height, width, channels)),
    keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer=initializer, bias_initializer=initializer),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer, bias_initializer=initializer),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer, bias_initializer=initializer),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer, bias_initializer=initializer),

    keras.layers.Flatten(),

    keras.layers.Dense(32, activation='relu', kernel_initializer=initializer, bias_initializer=initializer),
    keras.layers.Dense(classes, activation='softmax', kernel_initializer=initializer, bias_initializer=initializer),
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Check for GPUs
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print("GPUs detected:", gpus)
# else:
#     print("No GPUs detected.")
# exit(0)

start_time = time.time()

model.fit(images, labels, epochs=epochs, batch_size=batch_size)

predictions = model.predict(images)
end_time = time.time()

predicted_labels = np.argmax(predictions, axis=1)

print(predictions)
print(labels)
print(predicted_labels)

correct = 0

for true, predicted in zip(labels, predicted_labels):
    if true == predicted:
        correct += 1

print(f"Correct prediction = {correct}")

print(f"Total time = {end_time - start_time} seconds")
