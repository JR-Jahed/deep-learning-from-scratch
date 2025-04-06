import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

np.set_printoptions(linewidth=1000, suppress=True)

def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth   # (1, depth)

    angle_rates = 1 / (10000 ** depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

    return pos_encoding.astype(np.float32)

class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = torch.tensor(positional_encoding(length=2048, depth=d_model), dtype=torch.float32)

    def forward(self, x):
        length = x.shape[1]
        x = self.embedding(x)
        x *= np.sqrt(self.d_model)
        x = x + self.positional_encoding[torch.newaxis, :length, :]
        return x


class BaseAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        self.mha = nn.MultiheadAttention(num_heads=n_heads, embed_dim=d_model, batch_first=True)
        self.layer_normalisation = nn.LayerNorm(d_model)


class GlobalSelfAttention(BaseAttention):
    def forward(self, x):
        attention_output, _ = self.mha(
            query=x,
            value=x,
            key=x)
        x = torch.add(x, attention_output)
        x = self.layer_normalisation(x)
        return x


class CausalSelfAttention(BaseAttention):
    def forward(self, x):
        causal_mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), 1).bool()
        attention_output, _ = self.mha(
            query=x,
            value=x,
            key=x,
            attn_mask=causal_mask
        )
        x = torch.add(x, attention_output)
        x = self.layer_normalisation(x)
        return x


class CrossAttention(BaseAttention):
    def forward(self, x, context):
        attention_output, _ = self.mha(
            query=x,
            key=context,
            value=context,
        )
        x = torch.add(x, attention_output)
        x = self.layer_normalisation(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )
        self.layer_normalisation = nn.LayerNorm(d_model)

    def forward(self, x):
        x = torch.add(x, self.seq(x))
        x = self.layer_normalisation(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.self_attention = GlobalSelfAttention(n_heads=n_heads, d_model=d_model)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = self.self_attention(x)
        x = self.feed_forward(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.positional_embedding = PositionalEmbedding(vocab_size, d_model)
        self.encoder_layers = [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]

    def forward(self, x):
        x = self.positional_embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.causal_self_attention = CausalSelfAttention(n_heads=n_heads, d_model=d_model)
        self.cross_attention = CrossAttention(n_heads=n_heads, d_model=d_model)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x, context):
        x = self.causal_self_attention(x)
        x = self.cross_attention(x, context)
        x = self.feed_forward(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.positional_embedding = PositionalEmbedding(vocab_size, d_model)
        self.decoder_layers = [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]

    def forward(self, x, context):
        x = self.positional_embedding(x)
        for layer in self.decoder_layers:
            x = layer(x, context)

        return x


class Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, input_vocab_size, target_vocab_size):
        super().__init__()
        self.encoder = Encoder(d_model, n_layers, n_heads, d_ff, input_vocab_size)
        self.decoder = Decoder(d_model, n_layers, n_heads, d_ff, target_vocab_size)
        self.final_layer = nn.Linear(in_features=d_model, out_features=target_vocab_size)

    def forward(self, inputs):
        context, x = inputs
        context = self.encoder(context)

        x = self.decoder(x, context)

        logits = self.final_layer(x)
        return logits


if __name__ == '__main__':

    input_vocab_size = 10
    target_vocab_size = 15
    num_sequences = 5
    max_sequence_length = 10
    epochs = 50
    batch_size = 32

    d_model = 128
    n_layers = 6
    n_heads = 8
    d_ff = 512

    input_data = torch.randint(0, input_vocab_size, (num_sequences, max_sequence_length))
    target_data = torch.randint(0, target_vocab_size, (num_sequences, max_sequence_length))

    dataset = TensorDataset(input_data, target_data)

    loader = DataLoader(dataset, batch_size=batch_size)

    for i in range(num_sequences):
        print(input_data[i], "  ", target_data[i])
    print("\n\n")

    transformer = Transformer(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(transformer.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        total_loss = 0

        for input_batch, target_batch in loader:
            optimizer.zero_grad()
            logits = transformer((input_batch, target_batch))

            # Reshape for CrossEntropyLoss
            logits = logits.view(-1, target_vocab_size)
            target_batch = target_batch.view(-1)

            loss = loss_function(logits, target_batch)
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs}, Loss: {total_loss / num_sequences:.4f}")
    print("\n")

    logits = transformer((input_data, target_data))
    probabilities = nn.functional.softmax(logits, dim=-1).detach().numpy()

    for probabilities_ in probabilities:
        for p in probabilities_:
            print(p, "  --------   ", np.max(p))
        print("\n")
    print("\n\n")

    correct_generated_tokens = 0

    for i in range(num_sequences):
        tokens = []
        for j in range(max_sequence_length):
            token = np.argmax(probabilities[i][j])
            tokens.append(token)

            if token == target_data[i][j]:
                correct_generated_tokens += 1

        print(target_data[i], "  ", np.array(tokens))

    print(f"\nCorrect generated tokens: {correct_generated_tokens}  Accuracy: {(correct_generated_tokens / (num_sequences * max_sequence_length) * 100):.2f}%")