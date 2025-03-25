import numpy as np

np.set_printoptions(linewidth=1000)

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


"""
The shape of the input to MultiHeadAttention is (batch_size, max_sequence_length, d_model)
"""


if __name__ == '__main__':

    vocab_size = 10
    embedding_dim = 32
    num_sequences = 5
    max_sequence_length = 6
    epochs = 10
    batch_size = 32
    d_model = 4

    input_data = np.random.randint(0, vocab_size, (num_sequences, max_sequence_length))

    print(input_data.shape)

    position_embedding = PositionalEmbedding(vocab_size, d_model)

    x = position_embedding.forward(input_data)  # x shape is (batch_size, max_sequence_length, d_model)

    print(x.shape)

    print(x)
