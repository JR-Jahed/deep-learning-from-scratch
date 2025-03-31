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


class Dense:
    def __init__(self, input_channels, output_channels):
        self.input_channels = input_channels
        self.output_channels = output_channels
        limit = np.sqrt(6.0 / (input_channels + output_channels))
        self.weights = np.random.uniform(-limit, limit, (input_channels, output_channels))
        self.biases = np.random.uniform(-limit, limit, output_channels)

    def forward(self, x):
        """
        @Params
        x: (batch_size, max_sequence_length, d_model)
        @Returns
        output: (batch_size, max_sequence_length, d_model)
        """

        output = np.dot(x, self.weights) + self.biases
        return output


class FeedForward:
    def __init__(self, d_model, d_ff):
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

        output = self.dense1.forward(x)
        # Perform ReLU
        output = np.maximum(0, output)
        output = self.dense2.forward(output)
        output = self.add.forward([x, output])
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

    gsa = GlobalSelfAttention(n_heads=4, d_model=d_model)
    gsa_output = gsa.forward(x)
    print("gsa_output shape: ", gsa_output.shape)
    print(gsa_output, "\n\n")

    ff_encoder = FeedForward(d_model=d_model, d_ff=16)
    ff_encoder_output = ff_encoder.forward(gsa_output)
    print("ff_encoder_output shape: ", ff_encoder_output.shape)
    print(ff_encoder_output, "\n\n")

    csa = CausalSelfAttention(n_heads=4, d_model=d_model)
    csa_output = csa.forward(x)
    print("csa_output shape: ", csa_output.shape)
    print(csa_output, "\n\n")

    ca = CrossAttention(n_heads=4, d_model=d_model)
    ca_output = ca.forward(x=csa_output, context=ff_encoder_output)
    print("ca_output shape: ", ca_output.shape)
    print(ca_output, "\n\n")

    ff_decoder = FeedForward(d_model=d_model, d_ff=16)
    ff_decoder_output = ff_decoder.forward(csa_output)
    print("ff_decoder_output shape: ", ff_decoder_output.shape)
    print(ff_decoder_output, "\n\n")