import math
import numpy as np

def loss(predicted_probs, true_label):
    # Check that the true label is a valid class index
    if true_label < 0 or true_label >= len(predicted_probs):
        print("Error: Invalid class label")
        return -1

    if predicted_probs[true_label] <= 0:
        print("-------------------------------------------------------------------------------------||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

    return -math.log(predicted_probs[true_label])


class Dense:
    def __init__(self, input_length, neurons, activation):
        self.input_length = input_length
        self.neurons = neurons
        self.activation = activation
        self.weights = np.random.normal(0.0, .01, (input_length, neurons))  # Randomly initialized weights
        self.biases = np.full(neurons, 0.1)  # Initialize biases to 0.1


    def forward(self, input):

        # if len(input) == 16 and np.array(input).reshape(4, 4) in data:
        #     for i in range(len(data)):
        #         if np.array_equal(data[i][0], np.array(input).reshape(4, 4)):
        #             print("yes found it at  ", i)

        output = np.dot(input, self.weights) + self.biases

        # output = np.zeros(self.neurons)
        # # Linear combination (input * weights + biases)
        # for i in range(len(self.neurons)):
        #     for j in range(len(input)):
        #         output[i] += input[j] * self.weights[j][i]
        #     output[i] += self.biases[i]

        if max(output) > 100:
            print("output ------------------------------------------------")
            print(len(input), "    ", " ".join(map(str, input)))
            print(len(self.weights), "    ", " ".join(map(str, self.weights)))
            print(len(output), "    ", " ".join(map(str, output)))

        # Apply activation function
        self.activation_function(output)
        return output

    def backward(self, dL_dout, input, learning_rate):
        # print("dense backward")
        #
        # print(f"input size = {len(input)}   weights size = {len(self.weights)}  neurons = {self.neurons}")

        dL_din = np.zeros_like(input)  # Gradient w.r.t. input
        dL_dweights = np.zeros_like(self.weights)  # Gradient w.r.t. weights
        dL_dbiases = np.zeros_like(self.biases)  # Gradient w.r.t. biases

        # Gradients for weights and biases
        for i in range(self.neurons):
            dL_dbiases[i] = dL_dout[i]
            for j in range(len(input)):
                dL_dweights[j][i] += dL_dout[i] * input[j]
                dL_din[j] += self.weights[j][i] * dL_dout[i]

        # print(f"dL_din = {dL_din}")
        # print("dense backward done\n")

        for i in range(self.neurons):
            self.biases[i] -= learning_rate * dL_dbiases[i]
            for j in range(len(input)):
                self.weights[j][i] -= learning_rate * dL_dweights[j][i]

        # # Update weights and biases
        # for i in range(self.neurons):
        #     self.biases[i] -= learning_rate * dL_dbiases[i]
        #     self.weights[i] -= learning_rate * dL_dweights[i]

        return dL_din  # Pass gradient to the previous layer

    def activation_function(self, values):
        if self.activation == "relu":
            values[:] = np.maximum(values, 0.0)
        else:  # Softmax
            # print("line 79", values)
            exp_values = np.exp(values)
            sum_exp = np.sum(exp_values)
            values[:] = exp_values / sum_exp


class Sequential:
    def __init__(self):
        self.dense_layers = []  # List to store dense layers
        self.dense_outputs = []  # List to store outputs from dense layers

    def add_dense_layer(self, dense_layer):
        self.dense_layers.append(dense_layer)

    def fit(self, epochs, input_data, labels, learning_rate):
        predictions = []

        for epoch in range(1, epochs + 1):

            if np.max(input_data) > 1:
                print(f"\n\n\n\n----------------------------------\ninput data greater than 1-----------------------------\n\n\n\n")

            predictions.clear()
            for input_datum in input_data:
                predictions.append(self.forward(input_datum))

            print(f"Epoch = {epoch}    ", end="")
            # print("Predictions -----------")
            total_loss = 0

            for prediction, label in zip(predictions, labels):
                total_loss += loss(prediction, label)
                # print(*[f"{num:.12f}" for num in prediction])
                # print(" ".join(map(str, prediction)))

            print("Loss: ", total_loss, "\n")

            dL_dout = np.zeros_like(predictions)

            for i in range(len(predictions)):
                for j in range(len(predictions[0])):
                    dL_dout[i][j] = predictions[i][j] - (1.0 if j == labels[i] else 0.0)
                    # print(" ".join(map(str, dL_dout[i])), "\n")
                    # print(*[f"{num:.8f}" for num in dL_dout[i]])
                    self.backward(dL_dout[i], learning_rate, input_data[i])

        # for prediction in predictions:
        #     print(*[f"{num:.12f}" for num in prediction])

        return predictions

    def forward(self, input):
        self.dense_outputs.clear()

        flat_input = [value for matrix in input for row in matrix for value in row]

        #print(flat_input)

        # Forward pass through dense layers
        current_dense_input = flat_input
        for dense_layer in self.dense_layers:
            current_dense_input = dense_layer.forward(current_dense_input)
            self.dense_outputs.append(current_dense_input)

        return current_dense_input

    def backward(self, dL_dout, learning_rate, input_datum):
        # print("seq backward")
        current_dL_dout = dL_dout

        # Backward pass through dense layers (in reverse order)
        for i in range(len(self.dense_layers) - 1, -1, -1):
            current_input = self.flatten(input_datum) if i == 0 else self.dense_outputs[i - 1]
            current_dL_dout = self.dense_layers[i].backward(current_dL_dout, current_input, learning_rate)
        # print("dL size = ", len(current_dL_dout), " dense backward finished")

    def flatten(self, tensor):
        return [value for matrix in tensor for row in matrix for value in row]


if __name__ == "__main__":
    width = 4
    height = 4

    total_data = 100
    classes = 3

    data = np.random.randint(0, 256, (total_data, 1, width, height), dtype=np.uint8)
    data = data / 255.0

    labels = np.random.randint(0, classes, total_data)
    print(f"labels = {labels}")

    # Define the model (Sequential)
    model = Sequential()

    # Add the first dense layer with 32 neurons and ReLU activation
    dense1 = Dense(input_length=16, neurons=15, activation="relu")
    model.add_dense_layer(dense1)

    # Add the second dense layer with 3 neurons and Softmax activation
    dense2 = Dense(input_length=15, neurons=classes, activation="softmax")
    model.add_dense_layer(dense2)

    model.fit(epochs=100, input_data=data, labels=labels, learning_rate=0.01)
