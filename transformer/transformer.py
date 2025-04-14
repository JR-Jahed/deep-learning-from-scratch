import time
import numpy as np

np.set_printoptions(linewidth=10000, suppress=True)


def softmax(x, axis=-1):
    # Subtract max for numerical stability
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_entropy_loss(probabilities, target):
    """
    Computes the sequence-to-sequence loss.

    @Params:
    probabilities: (batch_size, max_sequence_length, target_vocab_size) - predicted scores
    target: (batch_size, max_sequence_length) - actual token indices

    @Returns:
    loss: Scalar loss value
    """

    batch_size, seq_length, vocab_size = probabilities.shape

    # Extract the probability of the correct token at each position
    correct_probabilities = probabilities[np.arange(batch_size)[:, None], np.arange(seq_length), target]

    # Compute the negative log-likelihood
    loss = -np.log(correct_probabilities)

    # Compute mean loss over all tokens
    return np.mean(loss)


def cross_entropy_gradient(probabilities, target):
    """
    @Params:
    probabilities: (batch_size, max_sequence_length, target_vocab_size) - predicted scores
    target: (batch_size, max_sequence_length) - actual token indices

    @Returns:
    gradients: (batch_size, max_sequence_length, target_vocab_size)
    """

    batch_size, seq_length, vocab_size = probabilities.shape
    gradient = probabilities.copy()
    gradient[np.arange(batch_size)[:, None], np.arange(seq_length), target] -= 1

    return gradient / batch_size


class Embedding:

    # d_model is embedding dimension
    # Let's call it d_model instead of embedding dimension for consistency

    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embeddings = np.random.uniform(0, 1, (vocab_size, d_model))
        self.input = None

    def forward(self, x):
        """
        @Params:
        x: (batch_size, max_sequence_length)
        @Returns:
        output: (batch_size, max_sequence_length, d_model)
        """
        self.input = x
        return self.embeddings[x]
    
    def backward(self, gradient_output, learning_rate):
        """
        @Params:
        gradient_output: (batch_size, max_sequence_length, d_model)
        learning_rate: float
        """
        
        unique_tokens = np.unique(self.input)  # Get unique token indices

        # Accumulate gradients for each unique token
        for token in unique_tokens:
            mask = self.input == token  # Mask for token occurrences
            self.embeddings[token] -= learning_rate * np.sum(gradient_output[mask], axis=0)


def positional_encoding(length, depth):
    """
    Will come back to it later to understand what's happening
    """

    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return pos_encoding.astype(np.float32)


class PositionalEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def forward(self, x):
        """
        @Param
        x: (batch_size, max_sequence_length)
        @Returns
        output: (batch_size, max_sequence_length, d_model)
        """
        length = x.shape[1]
        output = self.embedding.forward(x)
        output *= np.sqrt(float(self.d_model))
        output = output + self.pos_encoding[np.newaxis, :length, :]

        return output

    def backward(self, gradient_output, learning_rate):
        """
        @Param:
        gradient_output: (batch_size, max_sequence_length, d_model)
        learning_rate: float

        @Returns:
        None (updates the embedding weights)
        """
        # Backpropagate through the addition:
        # Since positional encoding is constant, its gradient is 0.
        # Only the scaled embedding part needs to propagate the gradient.

        # The forward pass did: output = sqrt(d_model) * embedding_output + constant
        # Thus, d(embedding_output)/d(output) = sqrt(d_model)
        gradient_embedding = gradient_output * np.sqrt(float(self.d_model))

        # Pass the gradient to the embedding layer's backward function.
        self.embedding.backward(gradient_embedding, learning_rate)


class ScaledDotProductAttention:
    def __init__(self, d_model, d):
        self.query = None
        self.key = None
        self.value = None
        self.Q = None
        self.K = None
        self.V = None
        self.A = None
        self.d_model = d_model
        self.d = d

        limit = np.sqrt(6 / (d_model + d))

        self.WQ = np.random.uniform(-limit, limit, (d_model, d))
        self.WK = np.random.uniform(-limit, limit, (d_model, d))
        self.WV = np.random.uniform(-limit, limit, (d_model, d))

    def forward(self, query, key, value, use_causal_mask=False):
        """
        @Params
        query: (batch_size, max_sequence_length, d_model)
        key: (batch_size, max_sequence_length, d_model)
        value: (batch_size, max_sequence_length, d_model)

        @Returns
        output: (batch_size, max_sequence_length, d)
        """

        self.query = query
        self.key = key
        self.value = value

        self.Q = np.dot(query, self.WQ)  # shape: (batch_size, max_sequence_length, d)
        self.K = np.dot(key, self.WK)  # shape: (batch_size, max_sequence_length, d)
        self.V = np.dot(value, self.WV)  # shape: (batch_size, max_sequence_length, d)

        # Compute scaled dot product
        QK = np.matmul(self.Q, self.K.transpose(0, 2, 1))  # shape: (batch_size, max_sequence_length, max_sequence_length)
        QK /= np.sqrt(self.d)

        # If causal mask is true, mask out future tokens for each token position.
        if use_causal_mask:
            # Create a mask: 1s in upper triangular (excluding main diagonal), then multiply by -inf.
            mask = np.triu(np.ones_like(QK), k=1) * (-1e9)
            QK = QK + mask

        # Apply softmax
        self.A = softmax(QK, axis=-1)

        output = np.matmul(self.A, self.V)

        return output  # shape: (batch_size, max_sequence_length, d)

    def backward(self, gradient_output, learning_rate):
        """
        @Param:
        gradient_output: (batch_size, max_sequence_length, d)
        learning_rate: float

        @Returns
        gradient_query, gradient_key, gradient_value: (batch_size, max_sequence_length, d_model)
        """

        print("scaled max: ", np.max(gradient_output), end="  ---  ")

        scale = 1 / np.sqrt(self.d)

        # --- Backprop through final multiplication: output = A . V ---
        # d_output/dV: gradient flows via V
        gradient_V = np.matmul(self.A.transpose(0, 2, 1), gradient_output)  # shape: (batch_size, max_sequence_length, d)
        print("V: ", np.max(gradient_V), end="  ---  ")

        # Gradient with respect to A:
        gradient_A = np.matmul(gradient_output, self.V.transpose(0, 2, 1))  # shape: (batch_size, max_sequence_length, max_sequence_length)
        print("A: ", np.max(gradient_A), end="  ---  ")

        # --- Backprop through softmax ---
        # For softmax, derivative is: dZ (dQK) = A * (dA - sum(dA * A, axis=-1, keepdims=True))
        sum_gradient_A_A = np.sum(gradient_A * self.A, axis=-1, keepdims=True)  # shape: (batch_size, max_sequence_length, 1)
        print("A_A: ", np.max(sum_gradient_A_A), end="  ---  ")
        gradient_QK = self.A * (gradient_A - sum_gradient_A_A)  # shape: (batch_size, max_sequence_length, max_sequence_length)
        print("QK: ", np.max(gradient_QK), end="  ---  ")

        # --- Backprop through scaling ---
        gradient_QK *= scale

        # --- Backprop through the dot product QK = Q . K^T ---
        # Gradient with respect to Q and K:
        gradient_Q = np.matmul(gradient_QK, self.K)  # shape: (batch_size, max_sequence_length, d)
        print("Q: ", np.max(gradient_Q), end="  ---  ")
        gradient_K = np.matmul(gradient_QK.transpose(0, 2, 1), self.Q)  # shape: (batch_size, max_sequence_length, d)
        print("K: ", np.max(gradient_K), end="  ---  ")

        # --- Backprop through linear projections ---
        # For Q = query . WQ, gradients:
        gradient_WQ = np.sum(np.matmul(self.query.transpose(0, 2, 1), gradient_Q), axis=0)  # shape: (d_model, d)
        print("WQ: ", np.max(gradient_WQ), end="  ---  ")
        gradient_query = np.matmul(gradient_Q, self.WQ.T)  # shape: (batch_size, max_sequence_length, d_model)
        print("query: ", np.max(gradient_query), end="  ---  ")

        # For K = key . WK:
        gradient_WK = np.sum(np.matmul(self.key.transpose(0, 2, 1), gradient_K), axis=0)  # shape: (d_model, d)
        print("WK: ", np.max(gradient_WK), end="  ---  ")
        gradient_key = np.matmul(gradient_K, self.WK.T)  # shape: (batch_size, max_sequence_length, d_model)
        print("key: ", np.max(gradient_key), end="  ---  ")

        # For V = value . WV:
        gradient_WV = np.sum(np.matmul(self.value.transpose(0, 2, 1), gradient_V), axis=0)  # shape: (d_model, d)
        print("WV: ", np.max(gradient_WV), end="  ---  ")
        gradient_value = np.matmul(gradient_V, self.WV.T)  # shape: (batch_size, max_sequence_length, d_model)
        print("value: ", np.max(gradient_value))

        self.WQ -= learning_rate * gradient_WQ
        self.WK -= learning_rate * gradient_WK
        self.WV -= learning_rate * gradient_WV

        return gradient_query, gradient_key, gradient_value


class MultiHeadAttention:
    def __init__(self, n_heads, d_model):
        self.concatenated_output = None
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        # Create a list of ScaledDotProductAttention layers for each head
        self.attention_heads = [ScaledDotProductAttention(d_model, self.head_dim) for _ in range(n_heads)]

        limit = np.sqrt(6 / (d_model + d_model))

        # Final linear projection matrix: shape (d_model, d_model)
        self.W = np.random.uniform(-limit, limit, (d_model, d_model))

    def forward(self, query, key, value, use_causal_mask=False):
        """
        @Params
        query, key, value: (batch_size, max_sequence_length, d_model)
        @Returns
        output: (batch_size, max_sequence_length, d_model)
        """

        head_outputs = []  # (n_heads, batch_size, max_sequence_length, d)

        for head in self.attention_heads:
            head_outputs.append(head.forward(query, key, value, use_causal_mask))  # Each head output: (batch_size, max_sequence_length, head_dim)

        # Concatenate along the last dimension
        self.concatenated_output = np.concatenate(head_outputs, axis=-1)  # shape: (batch_size, max_sequence_length, d_model)

        # Final linear projection
        final_output = np.dot(self.concatenated_output, self.W)  # shape: (batch_size, max_sequence_length, d_model)

        return final_output

    def backward(self, gradient_output, learning_rate):
        """
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradient_query, gradient_key, gradient_value: (batch_size, max_sequence_length, d_model)
        """

        # print("mha grad max: ", np.max(gradient_output), end="  ---  ")

        gradient_weights = np.dot(self.concatenated_output.reshape(-1, self.d_model).T, gradient_output.reshape(-1, self.d_model))

        # print("weights: ", np.max(gradient_weights), end="  ---  ")

        # Gradient with respect to the concatenated output
        gradient_concatenated_output = np.dot(gradient_output, self.W.T)

        # print("concat: ", np.max(gradient_concatenated_output), end="  ---  ")

        # Split gradient into each head's gradient
        gradient_heads = np.split(gradient_concatenated_output, self.n_heads, axis=-1)  # shape (batch_size, max_sequence_length, head_dim)

        gradient_query = 0
        gradient_key = 0
        gradient_value = 0
        print()
        # import os
        # import sys
        # original_stdout = sys.stdout
        # sys.stdout = open(os.devnull, 'w')

        for i, head in enumerate(self.attention_heads):
            gradient_q, gradient_k, gradient_v = head.backward(gradient_heads[i], learning_rate)
            gradient_query += gradient_q
            gradient_key += gradient_k
            gradient_value += gradient_v
        # sys.stdout = original_stdout

        self.W -= learning_rate * gradient_weights

        # print("query max: ", np.max(gradient_query), "    key max: ", np.max(gradient_key), "    value max: ", np.max(gradient_value))

        return gradient_query, gradient_key, gradient_value


class Add:
    def forward(self, x):
        """
        @Params
        x: a list of two matrices of shape (batch_size, max_sequence_length, d_model)
        """
        return x[0] + x[1]  # shape: (batch_size, max_sequence_length, d_model)

    def backward(self, gradient_output, learning_rate):
        """
        This function passes the same gradient it receives. It's not even necessary.
        Just added for consistency.

        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)
        @Returns
        gradients: a list of gradients corresponding to x[0] and x[1]
        """

        return [gradient_output, gradient_output]


class LayerNormalisation:
    def __init__(self, d_model, epsilon=1e-6):
        self.normalised = None
        self.standard_deviation = None
        self.variance = None
        self.mean = None
        self.g = np.ones(d_model)
        self.b = np.zeros(d_model)
        self.epsilon = epsilon

    def forward(self, x):
        """
        @Params
        x: (batch_size, max_sequence_length, d_model)
        @Returns
        output: (batch_size, max_sequence_length, d_model)
        """
        self.mean = np.mean(x, axis=-1, keepdims=True)  # shape: (batch_size, max_sequence_length, 1)
        self.variance = np.var(x, axis=-1, keepdims=True)  # shape: (batch_size, max_sequence_length, 1)
        self.standard_deviation = np.sqrt(self.variance + self.epsilon)  # shape: (batch_size, max_sequence_length, 1)

        self.normalised = (x - self.mean) / self.standard_deviation  # shape: (batch_size, max_sequence_length, d_model)

        output = self.g * self.normalised + self.b
        return output

    def backward(self, gradient_output, learning_rate):
        """
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)
        @Returns
        gradient_input: (batch_size, max_sequence_length, d_model)
        """

        gradient_g = np.sum(gradient_output * self.normalised, axis=(0, 1))  # shape: (d_model,)
        gradient_b = np.sum(gradient_output, axis=(0, 1))  # shape: (d_model,)

        gradient_normalised = gradient_output * self.g  # shape: (batch_size, max_sequence_length, d_model)

        # Following the formula for the gradient of layer norm:
        # d_x = (1/σ) * (d_normalized - mean(d_normalized) - normalized * mean(d_normalized * normalized))
        mean_gradient_normalised = np.mean(gradient_normalised, axis=-1, keepdims=True)  # shape: (batch_size, max_sequence_length, 1)
        mean_gradient_normalised_normalised = np.mean(gradient_normalised * self.normalised, axis=-1, keepdims=True)  # shape (batch_size, max_sequence_length, 1)

        gradient_input = (1.0 / self.standard_deviation) * (gradient_normalised - mean_gradient_normalised - self.normalised * mean_gradient_normalised_normalised)

        self.g -= learning_rate * gradient_g
        self.b -= learning_rate * gradient_b

        return gradient_input


class Dense:
    def __init__(self, input_channels, output_channels):
        self.input = None
        self.input_channels = input_channels
        self.output_channels = output_channels
        limit = np.sqrt(6.0 / (input_channels + output_channels))
        self.weights = np.random.uniform(-limit, limit, (input_channels, output_channels))
        self.biases = np.zeros(output_channels)

    def forward(self, x):
        """
        @Params
        x: (batch_size, max_sequence_length, input_channels)
        @Returns
        output: (batch_size, max_sequence_length, output_channels)
        """

        self.input = x

        output = np.dot(x, self.weights) + self.biases
        return output

    def backward(self, gradient_output, learning_rate):
        """
        @Params
        gradient_output: (batch_size, max_sequence_length, output_channels)
        @Returns
        gradient_input: (batch_size, max_sequence_length, input_channels)
        """

        gradient_input = np.dot(gradient_output, self.weights.T)

        weight_gradient = np.dot(gradient_output.reshape(-1, self.output_channels).T, self.input.reshape(-1, self.input_channels)).T

        bias_gradient = np.sum(gradient_output, axis=(0, 1))

        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient

        return gradient_input


class FeedForward:
    def __init__(self, d_model, d_ff):
        self.relu_mask = None
        self.d_model = d_model
        self.d_ff = d_ff
        self.dense1 = Dense(d_model, d_ff)
        self.dense2 = Dense(d_ff, d_model)
        self.add = Add()
        self.layer_normalisation = LayerNormalisation(d_model)

    def forward(self, x):
        """
        @Params
        x: (batch_size, max_sequence_length, d_model)
        @Returns
        output: (batch_size, max_sequence_length, d_model)
        """

        out1 = self.dense1.forward(x)  # shape: (batch_size, max_sequence_length, d_ff)
        relu_out = np.maximum(0, out1)  # ReLU activation
        self.relu_mask = (out1 > 0)  # Cache mask for ReLU
        out2 = self.dense2.forward(relu_out)  # shape: (batch_size, max_sequence_length, d_model)
        add_out = self.add.forward([x, out2])  # Residual addition
        final_out = self.layer_normalisation.forward(add_out)  # shape: (batch_size, max_sequence_length, d_model)
        return final_out

    def backward(self, gradient_output, learning_rate):
        """
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)
        @Returns
        gradient_input: (batch_size, max_sequence_length, d_model)
        """

        # 1. Backprop through Layer Normalisation
        gradient_add = self.layer_normalisation.backward(gradient_output, learning_rate)  # shape: (batch_size, max_sequence_length, d_model)

        # 2. Backprop through the addition layer: it splits the gradient equally to both inputs.
        gradient_x_res, gradient_out2 = self.add.backward(gradient_add, learning_rate)  # each of shape: (batch_size, max_sequence_length, d_model)

        # 3. Backprop through Dense2.
        # gradient_out2 is gradient with respect to Dense2 output.
        # dense2.backward returns gradient with respect to its input (i.e., relu_out)
        gradient_relu = self.dense2.backward(gradient_out2, learning_rate)  # shape: (batch_size, max_sequence_length, d_ff)

        # 4. Backprop through ReLU.
        # The derivative of ReLU is 1 for positive out1, 0 otherwise.
        gradient_out1 = gradient_relu * self.relu_mask  # shape: (batch_size, max_sequence_length, d_ff)

        # 5. Backprop through Dense1.
        # dense1.backward returns gradient with respect to its input x.
        gradient_x_dense = self.dense1.backward(gradient_out1, learning_rate)  # shape: (batch_size, max_sequence_length, d_model)

        gradient_input = gradient_x_res + gradient_x_dense  # shape: (batch_size, max_sequence_length, d_model)
        return gradient_input


class BaseAttention:
    def __init__(self, n_heads, d_model):
        self.n_heads = n_heads
        self.d_model = d_model
        self.mha = MultiHeadAttention(n_heads, d_model)
        self.add = Add()
        self.layer_normalisation = LayerNormalisation(d_model)


"""
This is the self-attention layer of encoder
"""
class GlobalSelfAttention(BaseAttention):
    def __init__(self, n_heads, d_model):
        super().__init__(n_heads, d_model)

    def forward(self, x):
        """
        @Params
        x: (batch_size, max_sequence_length, d_model)
        @Returns
        output: (batch_size, max_sequence_length, d_model)
        """
        attention_output = self.mha.forward(query=x, key=x, value=x)
        output = self.add.forward([x, attention_output])
        output = self.layer_normalisation.forward(output)
        return output

    def backward(self, gradient_output, learning_rate):
        """
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)
        @Returns
        gradient_x: (batch_size, max_sequence_length, d_model)
        """
        print("\nglobal", end="  ---  ")
        print("global grad max: ", np.max(gradient_output), end="  -----  ")

        # Gradient of the output of add layer
        gradient_add = self.layer_normalisation.backward(gradient_output, learning_rate)
        print("max layer-norm: ", np.max(gradient_add), end="  ----  ")

        """
        My assumption was correct. Not handling the gradients of residual connections was
        causing vanishing gradients. Even though some gradients are still becoming very small (1e-9),
        they are much larger than what they had become (1e-30) before taking the gradients of residual
        connections into account. Model consisting of more than one encoder-decoder layer is still
        unable to learn properly because of the small gradients. Let's try to solve it now.
        """

        # Gradient of input to this attention layer and gradient of output of MultiHeadAttention
        gradient_x, gradient_attention_output = self.add.backward(gradient_add, learning_rate)

        # As MultiHeadAttention receives 3 inputs (query, key, value), it returns 3 gradients during backpropagation
        gradient_query, gradient_key, gradient_value = self.mha.backward(gradient_attention_output, learning_rate)
        # print("query max: ", np.max(gradient_query), "    key max: ", np.max(gradient_key), "    value max: ", np.max(gradient_value))

        # Sum all the gradients to get the gradient of input which is the output of previous layer
        return gradient_x + gradient_query + gradient_key + gradient_value


"""
This is the self-attention layer of decoder
"""
class CausalSelfAttention(BaseAttention):
    def __init__(self, n_heads, d_model):
        super().__init__(n_heads, d_model)

    def forward(self, x):
        """
        @Params
        x: (batch_size, max_sequence_length, d_model)
        @Returns
        output: (batch_size, max_sequence_length, d_model)
        """
        attention_output = self.mha.forward(query=x, key=x, value=x, use_causal_mask=True)
        output = self.add.forward([x, attention_output])
        output = self.layer_normalisation.forward(output)
        return output

    def backward(self, gradient_output, learning_rate):
        """
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)
        @Returns
        gradient_x: (batch_size, max_sequence_length, d_model)
        """

        print("\ncausal", end="  ---  ")
        print("causal grad max: ", np.max(gradient_output), end="  -----  ")

        # Gradient of the output of add layer
        gradient_add = self.layer_normalisation.backward(gradient_output, learning_rate)
        print("max layer-norm: ", np.max(gradient_add), end="  ----  ")

        """
        My assumption was correct. Not handling the gradients of residual connections was
        causing vanishing gradients. Even though some gradients are still becoming very small (1e-9),
        they are much larger than what they had become (1e-30) before taking the gradients of residual
        connections into account. Model consisting of more than one encoder-decoder layer is still
        unable to learn properly because of the small gradients. Let's try to solve it now.
        """

        # Gradient of input to this attention layer and gradient of output of MultiHeadAttention
        gradient_x, gradient_attention_output = self.add.backward(gradient_add, learning_rate)

        # As MultiHeadAttention receives 3 inputs (query, key, value), it returns 3 gradients during backpropagation
        gradient_query, gradient_key, gradient_value = self.mha.backward(gradient_attention_output, learning_rate)
        # print("query max: ", np.max(gradient_query), "    key max: ", np.max(gradient_key), "    value max: ", np.max(gradient_value))

        # Sum all the gradients to get the gradient of input which is the output of previous layer
        return gradient_x + gradient_query + gradient_key + gradient_value


"""
This attention layer is part of decoder, but it connects encoder and decoder
"""
class CrossAttention(BaseAttention):
    def __init__(self, n_heads, d_model):
        super().__init__(n_heads, d_model)

    def forward(self, x, context):
        """
        @Params
        x: (batch_size, max_sequence_length, d_model)    comes from decoder
        context: (batch_size, max_sequence_length, d_model)    comes from encoder
        @Returns
        output: (batch_size, max_sequence_length, d_model)
        """
        attention_output = self.mha.forward(query=x, key=context, value=context)
        output = self.add.forward([x, attention_output])
        output = self.layer_normalisation.forward(output)
        return output
    
    def backward(self, gradient_output, learning_rate):
        """
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)
        @Returns
        gradient_x: (batch_size, max_sequence_length, d_model)
        gradient_context: (batch_size, max_sequence_length, d_model)
        """

        print("\ncross", end="  ----   ")
        print("cross grad max: ", np.max(gradient_output), end="  -----  ")

        # Gradient of the output of add layer
        gradient_add = self.layer_normalisation.backward(gradient_output, learning_rate)
        print("max layer-norm: ", np.max(gradient_add), end="  ----  ")

        """
        My assumption was correct. Not handling the gradients of residual connections was
        causing vanishing gradients. Even though some gradients are still becoming very small (1e-9),
        they are much larger than what they had become (1e-30) before taking the gradients of residual
        connections into account. Model consisting of more than one encoder-decoder layer is still
        unable to learn properly because of the small gradients. Let's try to solve it now.
        """

        # Gradient of input to this attention layer and gradient of output of MultiHeadAttention
        gradient_x, gradient_attention_output = self.add.backward(gradient_add, learning_rate)

        # As MultiHeadAttention receives 3 inputs (query, key, value), it returns 3 gradients during backpropagation
        gradient_query, gradient_key, gradient_value = self.mha.backward(gradient_attention_output, learning_rate)
        # print("query max: ", np.max(gradient_query), "    key max: ", np.max(gradient_key), "    value max: ", np.max(gradient_value))

        # Summing the gradients of x and query yields gradient of input
        # Gradient of x should only be added to the gradient of query because the residual connection is between
        # the output of the causal attention layer and the output of MultiHeadAttention of cross attention layer.
        # There's no residual connection that involves the output of encoder

        # Summing the gradients of key and value yields gradient of context
        return gradient_x + gradient_query, gradient_key + gradient_value


class EncoderLayer:
    def __init__(self, d_model, n_heads, d_ff):
        self.self_attention = GlobalSelfAttention(n_heads, d_model)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x):
        """
        @Params
        x: (batch_size, max_sequence_length, d_model)
        @Returns
        x: (batch_size, max_sequence_length, d_model)
        """

        x = self.self_attention.forward(x)
        x = self.feed_forward.forward(x)
        return x
    
    def backward(self, gradient_output, learning_rate):
        """
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)
        @Returns
        gradient: (batch_size, max_sequence_length, d_model)
        """

        # Gradient of the output of self attention layer
        gradient = self.feed_forward.backward(gradient_output, learning_rate)

        # Gradient of the input of self attention layer which was the output of
        # previous layer during forward pass
        gradient = self.self_attention.backward(gradient, learning_rate)
        return gradient


class Encoder:
    def __init__(self, d_model, n_layers, n_heads, d_ff, vocab_size):
        self.d_model = d_model
        self.n_layers = n_layers

        self.positional_embedding = PositionalEmbedding(vocab_size, d_model)
        self.encoder_layers = [EncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff) for _ in range(n_layers)]

    def forward(self, x):
        """
        @Params
        x: (batch_size, max_sequence_length)
        @Returns
        x: (batch_size, max_sequence_length, d_model)
        """
        x = self.positional_embedding.forward(x)  # shape: (batch_size, max_sequence_length, d_model)

        for layer in self.encoder_layers:
            x = layer.forward(x)

        return x

    def backward(self, gradient_output, learning_rate):
        """
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)
        @Returns
        gradient: (batch_size, max_sequence_length, d_model)
        """
        print("gradient max from decoder: ", np.max(gradient_output))

        gradient = gradient_output

        for layer in reversed(self.encoder_layers):
            gradient = layer.backward(gradient, learning_rate)
        
        self.positional_embedding.backward(gradient, learning_rate)
        
        return gradient


class DecoderLayer:
    def __init__(self, d_model, n_heads, d_ff):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.causal_self_attention = CausalSelfAttention(n_heads, d_model)
        self.cross_attention = CrossAttention(n_heads, d_model)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x, context):
        """
        @Params
        x: (batch_size, max_sequence_length, d_model)    comes from positional embedding of decoder for the first decoder layer, otherwise previous decoder layer
        context: (batch_size, max_sequence_length, d_model)    comes from encoder
        @Returns
        x: (batch_size, max_sequence_length, d_model)
        """

        x = self.causal_self_attention.forward(x)
        x = self.cross_attention.forward(x=x, context=context)
        x = self.feed_forward.forward(x)
        return x

    def backward(self, gradient_output, learning_rate):
        """
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)
        @Returns
        gradient_x_causal_self_attention: (batch_size, max_sequence_length, d_model)
        gradient_context: (batch_size, max_sequence_length, d_model)
        """

        # gradient_output is the gradient of the output of feed-forward layer of every decoder layer
        # gradient_cross_attention_output stores the gradient of the output of cross attention layer
        # which was the input of feed-forward during forward pass

        gradient_cross_attention_output = self.feed_forward.backward(gradient_output, learning_rate)

        # During forward pass, there were two inputs, x and context, which came from positional embedding
        # of decoder, and encoder respectively. Therefore, during backpropagation we calculate their gradients
        # and pass gradient of context to decoder layer which passes it to encoder because it came from
        # encoder, and gradient of x to positional embedding of decoder because it came from that layer.

        gradient_causal_self_attention_output, gradient_context = self.cross_attention.backward(gradient_cross_attention_output, learning_rate)

        # Gradient of input that goes into causal self-attention
        gradient_causal_self_attention_input = self.causal_self_attention.backward(gradient_causal_self_attention_output, learning_rate)

        return gradient_causal_self_attention_input, gradient_context


class Decoder:
    def __init__(self, d_model, n_layers, n_heads, d_ff, vocab_size):
        self.d_model = d_model
        self.n_layers = n_layers

        self.positional_embedding = PositionalEmbedding(vocab_size, d_model)
        self.decoder_layers = [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]

    def forward(self, x, context):
        """
        @Params
        x: (batch_size, max_sequence_length)
        context: (batch_size, max_sequence_length, d_model)    comes from encoder
        @Returns
        x: (batch_size, max_sequence_length, d_model)
        """

        x = self.positional_embedding.forward(x)  # shape: (batch_size, max_sequence_length, d_model)

        for layer in self.decoder_layers:
            x = layer.forward(x, context)

        return x

    def backward(self, gradient_output, learning_rate):
        """
        @Params
        gradient_output: (batch_size, max_sequence_length, d_model)

        @Returns
        gradient_context: (batch_size, max_sequence_length, d_model)
        """

        # During forward pass, for every decoder layer, x comes from previous decoder layer while
        # context is injected from outside, in this case, encoder. Therefore, gradient_x should be
        # passed to the previous layer during backpropagation.

        gradient_x = gradient_output

        # We need to pass this to encoder. It's the summation of the gradients of context of all the cross attention layers
        gradient_context = 0
        
        for layer in reversed(self.decoder_layers):
            gradient_x, gradient_context_ = layer.backward(gradient_x, learning_rate)
            gradient_context += gradient_context_

        self.positional_embedding.backward(gradient_x, learning_rate)

        return gradient_context


class Transformer:
    def __init__(self, d_model, n_layers, n_heads, d_ff, input_vocab_size, target_vocab_size):
        self.d_model = d_model
        self.n_layers = n_layers

        self.encoder = Encoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, vocab_size=input_vocab_size)
        self.decoder = Decoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, vocab_size=target_vocab_size)

        self.final_layer = Dense(d_model, target_vocab_size)

    def forward(self, inputs):
        """
        @Params
        inputs: A tuple of two sequences of shape (batch_size, max_sequence_length)
        @Returns
        logits: (batch_size, max_sequence_length, target_vocab_size)
        """

        context, x = inputs

        context = self.encoder.forward(context)

        x = self.decoder.forward(x, context)

        logits = self.final_layer.forward(x)

        return logits

    def backward(self, input_data, target_data, learning_rate):
        """
        @Params
        input_data: (batch_size, max_sequence_length)
        target_data: (batch_size, max_sequence_length)
        learning_rate: scalar
        @Returns
        loss: scalar
        """

        logits = self.forward((input_data, target_data))

        probabilities = softmax(logits)

        # loss = cross_entropy_loss(probabilities, target_data)

        gradient_for_final_layer = cross_entropy_gradient(probabilities, target_data)  # shape: (batch_size, max_sequence_length, target_vocab_size)
        # print("gradient_for_final_layer", gradient_for_final_layer.shape)
        # print(gradient_for_final_layer, "\n\n")

        gradient_for_decoder = self.final_layer.backward(gradient_for_final_layer, learning_rate)  # shape: (batch_size, max_sequence_length, d_model)
        # print("gradient_for_decoder: ", gradient_for_decoder.shape)
        # print(gradient_for_decoder, "\n\n")

        gradient_for_encoder= self.decoder.backward(gradient_for_decoder, learning_rate)  # shape: (batch_size, max_sequence_length, d_model)
        # print("gradient_for_encoder: ", gradient_for_encoder.shape)
        # print(gradient_for_encoder, "\n\n")

        self.encoder.backward(gradient_for_encoder, learning_rate)

        return 0

    def fit(self, input_data, target_data, epochs, batch_size, learning_rate=0.001, print_predictions=True):
        num_samples = len(input_data)

        for epoch in range(1, epochs + 1):
            print("\nEpoch: ", epoch)
            total_loss = 0

            for i in range(0, num_samples, batch_size):
                input_batch = input_data[i:i + batch_size]
                target_batch = target_data[i:i + batch_size]

                loss = self.backward(input_batch, target_batch, learning_rate)
                # total_loss += loss * input_batch.shape[0]

        #     if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
        #         print(f"Epoch {epoch:02d}/{epochs}, Loss: {total_loss / num_samples:.4f}")
        print("\n")

        predictions = []

        for i in range(num_samples):
            logits = self.forward((input_data[i].reshape(1, -1), target_data[i].reshape(1, -1))).squeeze(axis=0)
            prediction = softmax(logits)
            if print_predictions:
                for p in prediction:
                    print(p, " --------   ", np.max(p))
                print("\n")
            predictions.append(prediction)
        return predictions


if __name__ == '__main__':

    input_vocab_size = 10
    target_vocab_size = 15
    num_sequences = 10
    max_sequence_length = 10
    epochs = 2
    batch_size = 32

    # Model specifications
    n_layers = 6
    d_model = 128
    n_heads = 8
    d_ff = d_model * 4

    input_data = np.random.randint(0, input_vocab_size, (num_sequences, max_sequence_length))
    target_data = np.random.randint(0, target_vocab_size, (num_sequences, max_sequence_length))

    for i in range(num_sequences):
        print(input_data[i], "  ", target_data[i])
    print("\n\n")

    transformer = Transformer(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size)

    # import os
    # import sys
    # original_stdout = sys.stdout
    # sys.stdout = open(os.devnull, 'w')

    start_time = time.time()
    predictions = transformer.fit(input_data, target_data, epochs, batch_size, learning_rate=0.01, print_predictions=True)
    end_time = time.time()
    # sys.stdout = original_stdout

    correct_generated_tokens = 0

    for i in range(num_sequences):
        tokens = []
        for j in range(max_sequence_length):
            token = np.argmax(predictions[i][j])
            tokens.append(token)
            if target_data[i, j] == token:
                correct_generated_tokens += 1

        print(target_data[i], "  ", np.array(tokens))

    print(f"\nCorrect generated tokens: {correct_generated_tokens}  Accuracy: {(correct_generated_tokens / (num_sequences * max_sequence_length) * 100):.2f}%\n")
    print(f"Total time = {end_time - start_time} seconds")