"""
Train the same models using tensorflow and check how they fare
"""


import keras
import numpy as np
import time

np.set_printoptions(suppress=True, linewidth=1000, precision=5)


vocab_size = 1000
embedding_dim = 32
max_sequence_length = 10
num_sequences = 1000
num_classes = 5
epochs = 10
batch_size = 32

data = np.random.randint(0, vocab_size, (num_sequences, max_sequence_length))
labels = np.random.randint(0, num_classes, num_sequences)

train_percent = .8

X_train = data[:int(num_sequences * train_percent)]
y_train = labels[:int(num_sequences * train_percent)]

X_test = data[int(num_sequences * train_percent):]
y_test = labels[int(num_sequences * train_percent):]

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
    keras.layers.SimpleRNN(units=64, return_sequences=True),
    keras.layers.SimpleRNN(units=128, return_sequences=True),
    keras.layers.SimpleRNN(units=128, return_sequences=False),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

start_time = time.time()

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

predictions = model.predict(X_train)
end_time = time.time()

predicted_labels = np.argmax(predictions, axis=1)

for prediction in predictions:
    print(prediction, "    ---------    ", np.max(prediction))

# print(y_train)
# print(predicted_labels)

correct_prediction = 0
for predicted_label, correct_label in zip(predicted_labels, y_train):
    if predicted_label == correct_label:
        correct_prediction += 1

print(f"Correct prediction = {correct_prediction}  accuracy = {100 * correct_prediction/len(y_train)}")

print(f"Total time = {end_time - start_time} seconds")


# ----------------------------------------------------------------

print("\n\nTest:\n\n")

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

for prediction in predictions:
    print(prediction, "    ---------    ", np.max(prediction))

# print(y_test)
# print(np.array(predicted_labels))

correct_prediction = 0
for predicted_label, correct_label in zip(predicted_labels, y_test):
    if predicted_label == correct_label:
        correct_prediction += 1

print(f"\nCorrect prediction test = {correct_prediction}  accuracy = {100 * correct_prediction/len(y_test)}")
