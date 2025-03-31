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


class ScaledDotProductAttention:
    def __init__(self, d_model, d):

        self.d_model = d_model
        self.d = d

        self.WQ = np.random.uniform(0, 1, (d_model, d))
        self.WK = np.random.uniform(0, 1, (d_model, d))
        self.WV = np.random.uniform(0, 1, (d_model, d))

    def forward(self, query, key, value, use_causal_mask=False):

        """
        @Params
        query: (batch_size, max_sequence_length, d_model)
        key: (batch_size, max_sequence_length, d_model)
        value: (batch_size, max_sequence_length, d_model)

        @Returns
        output: (batch_size, max_sequence_length, d)
        """

        batch_size = key.shape[0]

        output = []  # List to store the output for each sample in the batch

        for i in range(batch_size):
            # Linear projections for Q, K, V
            Q = np.dot(query[i], self.WQ)  # shape: (max_sequence_length, d)
            K = np.dot(key[i], self.WK)  # shape: (max_sequence_length, d)
            V = np.dot(value[i], self.WV)  # shape: (max_sequence_length, d)

            # Compute scaled dot product
            QK = np.dot(Q, K.T)  # shape: (max_sequence_length, max_sequence_length)
            QK /= np.sqrt(self.d)

            # If causal mask is true, mask out future tokens for each token position.
            if use_causal_mask:
                # Create a mask: 1s in upper triangular (excluding main diagonal), then multiply by -inf.
                mask = np.triu(np.ones_like(QK), k=1) * (-1e9)
                QK = QK + mask

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

    def forward(self, query, key, value, use_causal_mask=False):

        """
        @Params
        x: (batch_size, max_sequence_length, d_model)
        @Returns
        output: (batch_size, max_sequence_length, d_model)
        """

        head_outputs = []  # (n_heads, batch_size, max_sequence_length, d)

        for head in self.attention_heads:
            head_outputs.append(head.forward(query, key, value, use_causal_mask))  # Each head output: (batch_size, max_sequence_length, head_dim)

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
    mha_output = mha.forward(x, x, x)
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