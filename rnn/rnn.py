import numpy as np
import time

"""

_________________________________________________________________________________________________________

This is the Python implementation of RNN.

IDE: PyCharm Professional (Downloaded using my Queen's email)


This is entirely for learning. There is no emphasis on test accuracy or optimisation. Implementation
of RNN from scratch without resorting to any library such as PyTorch or Tensorflow is the aim here.

_________________________________________________________________________________________________________

"""

np.set_printoptions(linewidth=1000, suppress=True)


def cross_entropy_loss(predictions, labels):
    """
    @Params
    predictions: (batch_size, num_classes)
    labels: (batch_size)

    @Returns
    loss: scalar
    """

    batch_size = predictions.shape[0]
    correct_probs = predictions[np.arange(batch_size), labels]
    loss = -np.sum(np.log(correct_probs + 1e-9))  # Avoid log(0)
    return loss / batch_size


def cross_entropy_gradient(predictions, labels):
    """
    @Params
    predictions: (batch_size, num_classes)
    labels: (batch_size)

    @Returns
    gradient: (batch_size, num_classes)
    """

    batch_size = predictions.shape[0]
    gradient = predictions.copy()
    gradient[np.arange(batch_size), labels] -= 1
    return gradient / batch_size


def orthogonal_init(size):
    # Generate a random matrix and perform QR decomposition
    a = np.random.randn(size, size)
    q, r = np.linalg.qr(a)
    # Ensure the matrix is orthogonal
    return q


class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.input = None

    def forward(self, x):
        """
        Fetch embeddings for a batch of token sequences.

        @Params
        x: (batch_size, sequence_length)

        @Returns
        embeddings: (batch_size, sequence_length, embedding_dim)
        """

        self.input = x  # Shape: (batch_size, sequence_length)
        return self.embeddings[x]  # Shape: (batch_size, sequence_length, embedding_dim)

    def backward(self, gradient_output, learning_rate):
        """
        Update embedding vectors using gradient.

        @Params
        gradient_output: (batch_size, sequence_length, embedding_dim).
        learning_rate: float
        """
        batch_size, sequence_length, _ = gradient_output.shape
        learning_rate *= batch_size
        unique_tokens = np.unique(self.input)  # Get unique token indices

        # Accumulate gradients for each unique token
        for token in unique_tokens:
            mask = self.input == token  # Mask for token occurrences
            self.embeddings[token] -= learning_rate * np.sum(gradient_output[mask], axis=0)


class SimpleRNN:
    def __init__(self, input_size, hidden_size, return_sequences=True):
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.input_size = input_size

        limit = np.sqrt(6 / (input_size + hidden_size))
        self.W_xh = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.W_hh = orthogonal_init(hidden_size)  # Shape: (hidden_size, hidden_size)
        self.b_h = np.zeros((hidden_size, 1))

    def forward(self, x):
        """
        @Params
        x: (batch_size, max_sequence_length, input_size)

        @Returns
        output: (batch_size, max_sequence_length, hidden_size) if return_sequences else (batch_size, hidden_size)
        """
        batch_size, max_sequence_length, _ = x.shape
        h = np.zeros((batch_size, self.hidden_size, 1))
        self.inputs = x
        self.h_states = []

        for t in range(max_sequence_length):
            # Select one timestep at a time from the input sequence
            x_t = x[:, t, :].reshape(batch_size, self.input_size, 1)  # Shape: (batch_size, input_size, 1)

            # Compute the hidden state for this timestep and drop the last dimension
            # The last dimension must be dropped to make the shape (hidden_size, batch_size) to be able
            # to add biases, which has shape (hidden_size, 1).

            h = np.tanh(np.dot(self.W_xh, x_t).squeeze(2) + np.dot(self.W_hh, h).squeeze(2) + self.b_h)  # Shape: (hidden_size, batch_size)
            h = np.expand_dims(h.T, -1)  # Shape: (batch_size, hidden_size, 1)
            self.h_states.append(h)

        self.h_states = np.array(self.h_states)  # Shape: (max_sequence_length, batch_size, hidden_size, 1)

        # Swap axes so that the first dimension corresponds to individual sequences in the batch
        self.h_states = np.swapaxes(self.h_states, 0, 1)  # Shape: (batch_size, max_sequence_length, hidden_size, 1)

        # Drop the last dimension to get the final output for each sequence in the batch
        output = self.h_states.squeeze(3)  # Shape: (batch_size, max_sequence_length, hidden_size)

        if self.return_sequences is False:
            # If return_sequences is False, then only the output for the last timestep is returned.
            output = output[:, -1, :]  # Shape: (batch_size, hidden_size)

        return output  # Shape: (batch_size, max_sequence_length, hidden_size) if return_sequences else (batch_size, hidden_size)

    def backward(self, dL_dh_last, learning_rate):
        """
        @Params
        dL_dh_last shape: (batch_size, max_sequence_length, hidden_size) if return_sequences else (batch_size, hidden_size)

        @Returns
        dL_dx: (batch_size, max_sequence_length, input_size)
        """
        batch_size, max_sequence_length, _ = self.inputs.shape
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db = np.zeros_like(self.b_h)

        dL_dx = np.zeros_like(self.inputs)

        if self.return_sequences:

            # dL_dh_next is the gradient of the hidden state of the next time step. Initially, it is zero since
            # there's no next time step for the last one.

            # If return_sequences is True, then the hidden state of each time step affected the final output
            # twice, once by being passed to the next layer and once by going through the next time step.
            # Therefore, the gradients of both these outputs should be summed to obtain the gradient of a
            # particular time step.

            dL_dh_next = np.zeros((batch_size, self.hidden_size))

            for t in reversed(range(max_sequence_length)):

                # Get the gradient of the loss for this time step by summing the gradient of
                # output and the gradient of the next time step.
                dL_dh_t = dL_dh_last[:, t, :] + dL_dh_next  # Shape: (batch_size, hidden_size)

                # Gradient of the input of activation function
                dtanh = (1 - self.h_states[:, t, :, 0] ** 2) * dL_dh_t  # Shape: (batch_size, hidden_size)

                # Gradient of the input to hidden weights
                dW_xh += np.dot(dtanh.T, self.inputs[:, t, :])  # Shape: (hidden_size, input_size)

                # Gradient of the hidden to hidden weights
                dW_hh += np.dot(dtanh.T, self.h_states[:, t - 1, :, 0] if t > 0 else np.zeros_like(self.h_states[:, t, :, 0]))  # Shape: (hidden_size, hidden_size)

                # Gradient of the biases
                db += np.sum(dtanh, axis=0, keepdims=True).T  # Shape: (hidden_size, 1)

                # Gradient of input
                dL_dx[:, t, :] = np.dot(dtanh, self.W_xh)  # Shape: (batch_size, input_size)

                # Gradient of hidden state for this time step which becomes next time step for the previous time step
                dL_dh_next = np.dot(dtanh, self.W_hh)  # Shape: (batch_size, hidden_size)
        else:
            # If return_sequences is False, then only the gradient of the last time step is used.
            # Because during forward pass, the hidden state of each time step before the last one
            # was used to calculate the hidden state of the next time step and affected the final
            # output through the next time step. But it didn't directly affect the output of next
            # layer as it wasn't passed to the next layer.

            dL_dh_t = dL_dh_last  # Shape: (batch_size, hidden_size)
            for t in reversed(range(max_sequence_length)):
                # Gradient of the input of activation function
                dtanh = (1 - self.h_states[:, t, :, 0] ** 2) * dL_dh_t  # Shape: (batch_size, hidden_size)

                # Gradient of the input to hidden weights
                dW_xh += np.dot(dtanh.T, self.inputs[:, t, :])  # Shape: (hidden_size, input_size)

                # Gradient of the hidden to hidden weights
                dW_hh += np.dot(dtanh.T, self.h_states[:, t - 1, :, 0] if t > 0 else np.zeros_like(self.h_states[:, t, :, 0]))  # Shape: (hidden_size, hidden_size)

                # Gradient of the biases
                db += np.sum(dtanh, axis=0, keepdims=True).T  # Shape: (hidden_size, 1)

                # Gradient of input
                dL_dx[:, t, :] = np.dot(dtanh, self.W_xh)  # Shape: (batch_size, input_size)

                # Gradient of hidden state for this time step
                dL_dh_t = np.dot(dtanh, self.W_hh)  # Shape: (batch_size, hidden_size)

        # Update parameters
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.b_h -= learning_rate * db

        return dL_dx


class Dense:
    def __init__(self, input_size, hidden_size, activation=None):
        self.hidden_size = hidden_size
        self.activation = activation

        limit = np.sqrt(6 / (input_size + hidden_size))
        self.W = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.b = np.zeros((hidden_size, 1))

    def forward(self, x):
        """Compute forward pass for a batch of inputs."""
        self.input = x  # Shape: (batch_size, input_size)
        z = np.dot(self.W, x.T) + self.b  # Shape: (hidden_size, batch_size)
        self.output = self.activation_function(z)  # Shape: (hidden_size, batch_size)
        return self.output.T  # Shape: (batch_size, hidden_size)

    def backward(self, gradient_output, learning_rate):
        """
        Compute gradients and update weights.

        @Params
        gradient_output: (batch_size, hidden_size).

        @Returns
        grad_input: (batch_size, input_size)
        """
        batch_size = gradient_output.shape[0]
        learning_rate *= batch_size

        dW = np.dot(gradient_output.T, self.input) / batch_size  # Shape: (hidden_size, input_size)
        db = np.sum(gradient_output.T, axis=1, keepdims=True) / batch_size

        gradient_input = np.dot(self.W.T, gradient_output.T).T  # Shape: (batch_size, input_size)

        # Update parameters
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        # Compute gradient for previous layer
        return gradient_input

    def activation_function(self, z):
        if self.activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        return z


class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, X, y_true, learning_rate=0.01):
        y_pred = self.forward(X)
        loss_grad = cross_entropy_gradient(y_pred, y_true)  # dL/dY

        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)

        loss = cross_entropy_loss(y_pred, y_true)
        return loss

    def fit(self, X_train, y_train, epochs, batch_size, learning_rate=0.01, print_predictions=True):

        num_samples = len(X_train)

        for epoch in range(1, epochs + 1):
            total_loss = 0

            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                loss = self.backward(X_batch, y_batch, learning_rate)
                total_loss += loss * X_batch.shape[0]

            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(f"Epoch {epoch:02d}/{epochs}, Loss: {total_loss / num_samples:.4f}")

        predictions = []

        for i in range(len(X_train)):
            prediction = self.forward(X_train[i].reshape(1, -1)).squeeze(axis=0)
            if print_predictions:
                print(prediction, " --------   ", np.max(prediction))
            predictions.append(prediction)
        return predictions

    def test(self, X_test):
        predictions = []

        for i in range(len(X_test)):
            prediction = self.forward(X_train[i].reshape(1, -1)).squeeze(axis=0)
            print(prediction, "   --------   ", np.max(prediction))
            predictions.append(prediction)

        return np.array(predictions)

if __name__ == "__main__":

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

    hidden_size = [64, 128, 128]

    # Define the model
    model = Sequential()
    model.add(Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim))

    for i in range(len(hidden_size)):
        model.add(SimpleRNN(input_size=hidden_size[i - 1] if i > 0 else embedding_dim,
                            hidden_size=hidden_size[i],
                            return_sequences=False if i == len(hidden_size) - 1 else True))

    model.add(Dense(input_size=hidden_size[-1], hidden_size=num_classes, activation='softmax'))

    start_time = time.time()
    # Train the model
    predictions = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, learning_rate=0.01, print_predictions=True)
    end_time = time.time()

    predicted_labels = np.argmax(predictions, axis=1)

    print(y_train)
    print(np.array(predicted_labels))

    correct_prediction = 0

    for predicted_label, correct_label in zip(predicted_labels, y_train):
        if predicted_label == correct_label:
            correct_prediction += 1

    print(f"Correct prediction train = {correct_prediction} accuracy = {100 * correct_prediction / len(y_train)}")
    print(f"Total time = {end_time - start_time} seconds\n\n")

    # ---------------------------------------------------------------------------------------------------------------------------------------

    print("\n\nTest:\n\n")

    predictions = model.test(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    print(y_test)
    print(np.array(predicted_labels))

    correct_prediction = 0
    for predicted_label, correct_label in zip(predicted_labels, y_test):
        if predicted_label == correct_label:
            correct_prediction += 1

    print(f"Correct prediction test = {correct_prediction}  accuracy = {100 * correct_prediction/len(y_test)}")