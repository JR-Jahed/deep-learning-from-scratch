import numpy as np

np.set_printoptions(linewidth=1000, suppress=True)


def softmax(x, axis=None):
    # Subtract max for numerical stability
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.input = None

    def forward(self, x):
        self.input = x
        return self.embeddings[x]


def positional_encoding(length, depth):

    """
    Will come back to it later to understand what's happening
    """

    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth   # (1, depth)

    angle_rates = 1 / (10000 ** depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

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
        length = x.shape[1]
        x = self.embedding.forward(x)
        x *= np.sqrt(float(self.d_model))
        x = x + self.pos_encoding[np.newaxis, :length, :]

        return x


class ScaledDotProductAttention:
    def __init__(self, d_model, d):

        self.d_model = d_model
        self.d = d

        self.WQ = np.random.uniform(0, 1, (d_model, d))
        self.WK = np.random.uniform(0, 1, (d_model, d))
        self.WV = np.random.uniform(0, 1, (d_model, d))

    def forward(self, x):

        """
        @Params
        x: (batch_size, max_sequence_length, d_model)

        @Returns
        output: (batch_size, max_sequence_length, d)
        """

        output = []  # List to store the output for each sample in the batch

        for mat in x:  # 'mat' shape: (max_sequence_length, d_model)
            # Linear projections for Q, K, V
            Q = np.dot(mat, self.WQ)  # shape: (max_sequence_length, d)
            K = np.dot(mat, self.WK)  # shape: (max_sequence_length, d)
            V = np.dot(mat, self.WV)  # shape: (max_sequence_length, d)

            # Compute scaled dot product
            QK = np.dot(Q, K.T)  # shape: (max_sequence_length, max_sequence_length)
            QK /= np.sqrt(self.d)

            # Mask (optional)

            # Apply softmax
            QK = softmax(QK, axis=-1)

            QKV = np.dot(QK, V)  # shape: (max_sequence_length, d)

            output.append(QKV)

        return np.array(output)  # shape: (batch_size, max_sequence_length, d)


class MultiHeadAttention:
    def __init__(self, n_heads, d_model):
        self.n_heads = n_heads
        self.d_model = d_model

        # Create a list of ScaledDotProductAttention layers for each head
        self.attention_heads = [ScaledDotProductAttention(d_model, d_model // n_heads) for _ in range(n_heads)]

        # Final linear projection matrix: shape (d_model, d_model)
        self.W = np.random.uniform(0, 1, (d_model, d_model))

    def forward(self, x):

        """
        @Params
        x: (batch_size, max_sequence_length, d_model)
        @Returns
        output: (batch_size, max_sequence_length, d_model)
        """

        head_outputs = []  # (n_heads, batch_size, max_sequence_length, d)

        for head in self.attention_heads:
            head_outputs.append(head.forward(x))  # Each head output: (batch_size, max_sequence_length, head_dim)

        # Concatenate along the last dimension
        concatenated = np.concatenate(head_outputs, axis=-1)  # shape: (batch_size, max_sequence_length, d_model)

        # Final linear projection
        output = np.dot(concatenated, self.W)  # shape: (batch_size, max_sequence_length, d_model)

        return output


class Add:
    def forward(self, x):
        """
        @Params
        x: a list of two matrices of shape (batch_size, max_sequence_length, d_model)
        """
        return x[0] + x[1]  # shape: (batch_size, max_sequence_length, d_model)


class LayerNormalisation:

    def __init__(self, d_model, epsilon=1e-6):
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
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        standard_deviation = np.sqrt(variance + self.epsilon)

        output = self.g * (x - mean) / standard_deviation + self.b

        return output



if __name__ == '__main__':

    vocab_size = 10
    embedding_dim = 32
    num_sequences = 5
    max_sequence_length = 6
    epochs = 10
    batch_size = 32
    d_model = 8

    input_data = np.random.randint(0, vocab_size, (num_sequences, max_sequence_length))

    position_embedding = PositionalEmbedding(vocab_size, d_model)

    x = position_embedding.forward(input_data)  # x shape is (batch_size, max_sequence_length, d_model)
    print("x shape: ", x.shape)
    print(x, "\n\n")

    mha = MultiHeadAttention(n_heads=4, d_model=d_model)
    mha_output = mha.forward(x)
    print("mha_output shape: ", mha_output.shape)
    print(mha_output, "\n\n")

    add = Add()
    add_output = add.forward([x, mha_output])
    print("add_output shape: ", add_output.shape)
    print(add_output, "\n\n")

    layer_normalisation = LayerNormalisation(d_model)
    layer_normalisation_output = layer_normalisation.forward(add_output)
    print("layer_normalisation_output shape: ", layer_normalisation_output.shape)
    print(layer_normalisation_output, "\n\n")