import numpy as np
import time
import sys
import os

"""

_________________________________________________________________________________________________________

This is the Python implementation of CNN.

IDE: PyCharm Professional (Downloaded using my Queen's email)


This is entirely for learning. There is no emphasis on test accuracy or optimisation. Implementation
of CNN from scratch without resorting to any library such as PyTorch or Tensorflow is the aim here.

_________________________________________________________________________________________________________

"""

def cross_entropy_loss(predictions, labels):
    """
    Compute the cross-entropy loss for integer labels.

    @Params
    predictions: (batch_size, num_classes)
    labels: (batch_size,)

    @Returns
    loss: Scalar loss value
    """
    batch_size = predictions.shape[0]
    # Extract the probabilities corresponding to the true labels
    correct_probs = predictions[np.arange(batch_size), labels]
    # Compute the loss
    return -np.sum(np.log(correct_probs + 1e-15)) / batch_size  # Add epsilon for numerical stability

def cross_entropy_gradient(predictions, labels):
    """
    Compute the gradient of the cross-entropy loss w.r.t. logits (softmax outputs).

    @Params
    predictions: (batch_size, num_classes)
    labels: (batch_size,)

    @Returns
    gradient: (batch_size, num_classes)
    """
    batch_size = predictions.shape[0]
    gradient = predictions.copy()
    gradient[np.arange(batch_size), labels] -= 1  # Subtract 1 from the correct class probabilities
    return gradient / batch_size



class Conv2d:
    def __init__(self, input_channels, output_channels, kernel_size = (3, 3), initializer = "glorot"):
        self.input = None
        self.relu_mask = None
        self.input_channels = input_channels
        self.output_channels = output_channels  # Number of filters
        self.kernel_size = kernel_size
        self.initializer = initializer


        if initializer == "glorot":
            # Glorot (Xavier) Uniform Initialization
            limit = np.sqrt(6.0 / (input_channels + output_channels))
            self.weights = np.random.uniform(-limit, limit, (output_channels, input_channels, kernel_size[0], kernel_size[1]))
            self.biases = np.random.uniform(-limit, limit, output_channels)
        elif initializer == "he":
            # He Uniform Initialization
            limit = np.sqrt(6.0 / input_channels)
            self.weights = np.random.uniform(-limit, limit, (output_channels, input_channels, kernel_size[0], kernel_size[1]))
            self.biases = np.random.uniform(-limit, limit, output_channels)
        else:
            # Initialize kernels (filters) and biases
            self.weights = np.random.normal(0.0, 0.01, (output_channels, input_channels, kernel_size[0], kernel_size[1]))
            self.biases = np.random.normal(0.0, 0.01, output_channels)


    def forward(self, input):

        """
        @Params
        input: (batch_size, input_height, input_width, input_channels)

        @Returns
        output: (batch_size, input_height, input_width, output_channels)
        """

        # Save input for backpropagation
        self.input = input

        batch_size, input_height, input_width, input_channels = input.shape

        # Calculate the output dimensions
        kernel_height, kernel_width = self.kernel_size

        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1

        output = np.zeros((batch_size, output_height, output_width, self.output_channels))

        for i in range(batch_size):
            for h in range(output_height):   # Loop over the height of the output
                for w in range(output_width):   # Loop over the width of the output
                    for oc in range(self.output_channels):   # Loop over the output channels
                        # Initialize the sum for this particular output location
                        sum_value = 0.0


                        """

                        For each position of the kernel on the input channels, the output for that particular position is the sum of
                        elementwise multiplication for all input channels. The kernel starts to slide over the input from [0, 0].
                        If there are 32 input channels, the output will be calculated as the sum of the hadamard product when kernel
                        is placed on the same position for all 32 channels.

                        """

                        for ic in range(input_channels):   # Loop over each input channel
                            # Extract the region of the input image that corresponds to the filter
                            for kh in range(kernel_height):   # Loop over the kernel height
                                for kw in range(kernel_width):   # Loop over the kernel width
                                    input_value = input[i, h + kh, w + kw, ic]
                                    kernel_value = self.weights[oc, ic, kh, kw]

                                    # Multiply the input value with the kernel value and add to the sum
                                    sum_value += input_value * kernel_value
                        output[i, h, w, oc] = sum_value + self.biases[oc]

        # ReLU Activation: Apply activation and save mask
        self.relu_mask = (output > 0)  # Save mask for backward pass
        output = np.maximum(0, output)  # Apply ReLU
        return output  # shape: (batch_size, input_height, input_width, output_channels)

    def backward(self, dL_dout, learning_rate):

        """
        @Params
        dL_dout: (batch_size, output_height, output_width, output_channels)
        learning_rate: A float

        @Returns
        dL_din: (batch_size, input_height, input_width, input_channels)
        """

        batch_size, input_height, input_width, input_channels = self.input.shape
        kernel_height, kernel_width = self.kernel_size

        # Initialize gradients for weights, biases, and input
        dL_dweights = np.zeros_like(self.weights)
        dL_dbiases = np.zeros_like(self.biases)
        dL_din = np.zeros_like(self.input)

        # Apply ReLU mask to dL_dout
        dL_dout *= self.relu_mask

        """

        First loop goes through all the images in the batch
        Second loop goes through the rows of the activation maps of this layer
        Third loop goes through the columns of the activation maps of this layer
        Fourth loop goes through the output channels of this layer
        Fifth loop goes through the input channels of this layer
        Sixth and Seventh loops go through the kernel height and width respectively

        """

        # Calculate gradients for weights, biases, and input
        for i in range(batch_size):
            for h in range(dL_dout.shape[1]):  # Output height
                for w in range(dL_dout.shape[2]):  # Output width
                    for oc in range(self.output_channels):  # Output channels

                        """

                        The bias value of a neuron was used to calculate all the output values of an activation map.
                        Therefore, its gradient is calculated by summing the gradients of all the output values.

                        """

                        dL_dbiases[oc] += dL_dout[i, h, w, oc]

                        # Gradient w.r.t. the weights (input region * output gradient)
                        for ic in range(input_channels):
                            for kh in range(kernel_height):
                                for kw in range(kernel_width):
                                    input_value = self.input[i, h + kh, w + kw, ic]

                                    """

                                    During forward pass kernel slid over all the possible positions of the input channels.

                                    Therefore, the gradient of a weight (dL_dweights[oc][ic][kh][kw]) will be calculated using the gradients of all
                                    the outputs that were calculated using this weight and its corresponding input value

                                    In other words, the gradient of a particular output position (dL_dout[i][h][w][oc]) will be used to calculate the
                                    gradients of all the weights that were used to calculate this particular output position

                                    """

                                    dL_dweights[oc, ic, kh, kw] += input_value * dL_dout[i, h, w, oc]

                                    """

                                    Similarly, the gradient of a particular input position (dL_din[i][h + kh][w + kw][ic]) will be calculated using the
                                    weight (weights[oc][ic][kh][kw]) associated with that input position and the gradient of the output position which
                                    was calculated using this input value

                                    In other words, gradient of every output value (dL_dout[i][h][w][oc]) will be used to calculate the gradient of all
                                    the input values that were used to calculate this output value

                                    """

                                    dL_din[i, h + kh, w + kw, ic] += self.weights[oc, ic, kh, kw] * dL_dout[i, h, w, oc]

        # Update the weights and biases
        self.weights -= learning_rate * dL_dweights
        self.biases -= learning_rate * dL_dbiases
        return dL_din  # shape: (batch_size, input_height, input_width, input_channels)


class Dense:
    def __init__(self, input_channels, output_channels, activation, initializer="glorot"):
        self.input = None
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.activation = activation
        self.initializer = initializer

        if initializer == "glorot":
            limit = np.sqrt(6.0 / (input_channels + output_channels))
            self.weights = np.random.uniform(-limit, limit, (input_channels, output_channels))
            self.biases = np.random.uniform(-limit, limit, output_channels)
        elif initializer == "he":
            limit = np.sqrt(6.0 / input_channels)
            self.weights = np.random.uniform(-limit, limit, (input_channels, output_channels))
            self.biases = np.random.uniform(-limit, limit, output_channels)
        else:
            self.weights = np.random.normal(0.0, .01, (input_channels, output_channels))  # Randomly initialized weights
            self.biases = np.random.normal(0.0, .01, output_channels)


    def forward(self, input):
        """
        @Params
        input: (batch_size, input_channels)

        @Returns
        output: (batch_size, output_channels).
        """

        # Save input for backpropagation
        self.input = input
        output = []

        for row in input:
            row_output = np.dot(row, self.weights) + self.biases
            output.append(row_output)

        output = np.array(output)
        output = self.activation_function(output)
        return output

    def backward(self, dL_dout, learning_rate):

        """
        @Params
        dL_dout: (batch_size, output_channels)
        learning_rate: float

        @Returns
        dL_din: (batch_size, input_channels)
        """

        batch_size = self.input.shape[0]
        dL_din = np.zeros_like(self.input)  # Gradient w.r.t. input
        dL_dweights = np.zeros_like(self.weights)  # Gradient w.r.t. weights
        dL_dbiases = np.zeros_like(self.biases)  # Gradient w.r.t. biases

        # Gradients for weights and biases
        for i in range(batch_size):
            for oc in range(self.output_channels):
                dL_dbiases[oc] += dL_dout[i][oc]
                for ic in range(self.input.shape[1]):
                    # Calculate gradients of weights
                    dL_dweights[ic][oc] += dL_dout[i][oc] * self.input[i][ic]

                    # Calculate gradients of inputs (outputs of the previous layer) to this layer
                    dL_din[i][ic] += self.weights[ic][oc] * dL_dout[i][oc]

        dL_dweights /= batch_size
        dL_dbiases /= batch_size

        # Update weights and biases
        for oc in range(self.output_channels):
            self.biases[oc] -= learning_rate * dL_dbiases[oc]
            for ic in range(self.input.shape[1]):
                self.weights[ic][oc] -= learning_rate * dL_dweights[ic][oc]

        return dL_din  # Pass gradient to the previous layer


    def activation_function(self, values):
        """
        Apply the activation function.
        """
        if self.activation == "relu":
            return np.maximum(0, values)
        elif self.activation == "softmax":
            exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))  # Numerical stability
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)
        else:
            raise ValueError("Unsupported activation function: " + self.activation)



class MaxPooling2D:

    def __init__(self):
        self.input = None


    def forward(self, input):
        """
        @Params
        input: (batch_size, height, width, channels)

        @Returns
        output: (batch_size, height / 2, width / 2, channels)
        """

        # Save input for backpropagation
        self.input = input

        batch_size, height, width, channels = input.shape

        # Calculate the output dimensions
        output_height = height // 2
        output_width = width // 2

        # Initialize the output
        output = np.zeros((batch_size, output_height, output_width, channels))

        for i in range(batch_size):
            for j in range(output_height):  # Loop through the output height
                for k in range(output_width):  # Loop through the output width
                    for l in range(channels):  # Loop through the channels
                        # Define the 2x2 region for pooling
                        region = input[i, j*2:j*2+2, k*2:k*2+2, l]

                        # Get the maximum value in the region
                        output[i, j, k, l] = np.max(region)
        return output


    def backward(self, dL_dout, learning_rate):

        """
        @Params
        dL_dout: (batch_size, output_height, output_width, channels)

        @Returns
        dL_din: (batch_size, input_height, input_width, channels)
        """

        batch_size, input_height, input_width, input_channels = self.input.shape

        # Initialize gradients for input
        dL_din = np.zeros_like(self.input)

        # Calculate gradients of input
        for i in range(batch_size):
            for h in range(dL_dout.shape[1]):   # Output height
                for w in range(dL_dout.shape[2]):   # Output width
                    for oc in range(dL_dout.shape[3]):   # Output channels = Input channels

                        # Find the 2x2 region in the input corresponding to this output
                        region = self.input[i, h * 2:h * 2 + 2, w * 2:w * 2 + 2, oc]
                        # Find the index of the max value in the region
                        max_index = np.unravel_index(np.argmax(region), region.shape)
                        # The gradient is only propagated to the maximum element
                        dL_din[i, h * 2 + max_index[0], w * 2 + max_index[1], oc] += dL_dout[i, h, w, oc]
        return dL_din


class Sequential:
    def __init__(self):
        self.conv_pool_layers = []  # List to store convolutional and pooling layers
        self.dense_layers = []  # List to store dense layers
        self.conv_pool_outputs = []  # List to store outputs from convolutional layers
        self.dense_outputs = []  # List to store outputs from dense layers
        self.last_conv_output_shape = None

    def add_conv_pool_layer(self, layer):
        self.conv_pool_layers.append(layer)

    def add_dense_layer(self, dense_layer):
        self.dense_layers.append(dense_layer)

    def fit(self, epochs, input_images, labels, batch_size, learning_rate):

        for epoch in range(1, epochs + 1):
            original_stdout = sys.stdout
            # Disable print
            # sys.stdout = open(os.devnull, 'w')
            total_loss = 0
            for i in range(len(input_images)):
                start_index = i * batch_size
                end_index = min(start_index + batch_size, len(input_images))
                if start_index >= len(input_images):
                    break

                batch_predictions = self.forward(input_images[start_index: end_index])
                dL_dout = cross_entropy_gradient(batch_predictions, labels[start_index: end_index])
                self.backward(dL_dout, learning_rate)
                total_loss += cross_entropy_loss(batch_predictions, labels[start_index: end_index]) * (end_index - start_index + 1)

            sys.stdout = original_stdout

            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(f"Epoch = {epoch:02}     Loss: {total_loss / len(input_images)}")

        print("\n")
        print("Predictions\n")

        predictions = self.forward(input_images)

        for prediction in predictions:
            print(*[f"{num:.12f}" for num in prediction])

        print("\n")

        return predictions

    def forward(self, input):
        self.conv_pool_outputs.clear()
        self.dense_outputs.clear()

        current_input = input

        # Forward pass through convolutional layers
        for layer in self.conv_pool_layers:
            current_input = layer.forward(current_input)

            # Cache the output of each convolutional layer
            self.conv_pool_outputs.append(current_input)

        self.last_conv_output_shape = current_input.shape

        # Flatten the output from the last convolutional layer
        # There should be one row for each image in the batch
        flat_input = [self.flatten(tensor) for tensor in current_input]

        # Forward pass through dense layers
        current_dense_input = np.array(flat_input)
        for dense_layer in self.dense_layers:
            current_dense_input = dense_layer.forward(current_dense_input)

            # Cache the output of each dense layer
            self.dense_outputs.append(current_dense_input)

        return current_dense_input

    def backward(self, dL_dout, learning_rate):
        current_dL_dout = dL_dout

        # Backward pass through dense layers (in reverse order)
        for i in range(len(self.dense_layers) - 1, -1, -1):
            current_dL_dout = self.dense_layers[i].backward(dL_dout=current_dL_dout, learning_rate=learning_rate)

        # Backward pass through convolutional layers (in reverse order)
        current_dL_dout_conv = self.unflatten(current_dL_dout)

        for i in range(len(self.conv_pool_layers) - 1, -1, -1):
            current_dL_dout_conv = self.conv_pool_layers[i].backward(dL_dout=current_dL_dout_conv, learning_rate=learning_rate)

    def flatten(self, tensor):
        return [value for matrix in tensor for row in matrix for value in row]

    def unflatten(self, flat):

        """
        get the output of the last convolutional layer
        """

        tensor = np.zeros(self.last_conv_output_shape)

        for i in range(self.last_conv_output_shape[0]):
            index = 0
            for j in range(self.last_conv_output_shape[1]):
                for k in range(self.last_conv_output_shape[2]):
                    for l in range(self.last_conv_output_shape[3]):
                        tensor[i, j, k, l] = flat[i][index]
                        index += 1
        return tensor


if __name__ == "__main__":

    np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.7f}'})
    width = 32
    height = 32
    channels = 3

    total_images = 1
    classes = 5
    epochs = 1
    batch_size = 32

    # Create images with random pixel values between 0 and 255
    images = np.random.randint(0, 256, (total_images, height, width, channels), dtype=np.uint8)
    images = images / 255.0  # Normalise the images to [0, 1]

    # Label for the image
    labels = np.random.randint(0, classes, total_images)
    # labels = np.array([3, 1, 4, 2, 0, 1, 4, 3, 2, 0])
    # labels = np.array([3, 3, 3, 3, 3])

    model = Sequential()

    trainable_parameters = 0

    conv_output_channels = [16, -1, 32, -1, 64]

    conv = None

    current_height = height
    current_width = width

    for i in range(len(conv_output_channels)):
        # If it's a positive value, add a convolutional layer. Otherwise, add a pooling layer
        if conv_output_channels[i] > 0:
            conv = Conv2d(input_channels=channels if i == 0 else conv.output_channels, output_channels=conv_output_channels[i])
            model.add_conv_pool_layer(conv)
            trainable_parameters += conv.weights.size + conv.biases.size
            current_height -= 2
            current_width -= 2
        else:
            model.add_conv_pool_layer(MaxPooling2D())
            current_height = current_height // 2
            current_width = current_width // 2

        if current_height <= 0 or current_width <= 0:
            raise ValueError(f"Height or Width cannot be non-positive    height = {current_height}  width = {current_width}")

    dense1 = Dense(input_channels=current_height * current_width * conv.output_channels, output_channels=32, activation="relu")
    model.add_dense_layer(dense1)

    dense2 = Dense(input_channels=32, output_channels=classes, activation="softmax")
    model.add_dense_layer(dense2)

    for dense_layer in model.dense_layers:
        trainable_parameters += dense_layer.weights.size + dense_layer.biases.size

    print("Total trainable parameters = ", trainable_parameters)

    original_stdout = sys.stdout
    # Disable print
    # Was printing in many places to debug while implementing. It became tedious to enable and disable at all these different places.
    # This one line disables all the subsequent print statements
    # sys.stdout = open(os.devnull, 'w')

    start_time = time.time()
    predictions = model.fit(epochs=epochs, input_images=images, labels=labels, batch_size=batch_size, learning_rate=0.01)
    end_time = time.time()

    # Restore original stdout
    sys.stdout = original_stdout

    predicted_labels = np.argmax(predictions, axis=1)

    print("correct labels =     ", labels)
    print("predicted labels =   ", predicted_labels)

    correct_prediction = 0

    for predicted_label, correct_label in zip(predicted_labels, labels):
        if predicted_label == correct_label:
            correct_prediction += 1

    print(f"Correct prediction = {correct_prediction}")

    print(f"Total time = {end_time - start_time} seconds")
