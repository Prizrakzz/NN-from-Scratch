import torch
import torch.nn as nn
import math
sentences = [
    "i like cats",
    "i like dogs",
    "you like cats",
    "cats eat fish"
]

tokenized_sentences = [sentence.split() for sentence in sentences]



all_tokens = []

for sentence in tokenized_sentences:
    for token in sentence:
        all_tokens.append(token)

vocab = sorted(set(all_tokens))

token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}





sentencetwo = "i like cats"

tokens = sentencetwo.split()
token_ids = [token_to_id[token] for token in tokens]



x = torch.tensor(token_ids)


vocab_size = len(vocab)
embedding_dim = 4
torch.manual_seed(0)

embedding_table = torch.randn(vocab_size, embedding_dim)



embeddings = embedding_table[x]


max_seq_length = 3

position_embedding_table = torch.randn(max_seq_length, embedding_dim)

position_ids = torch.arange(max_seq_length)

position_embeddings = position_embedding_table[position_ids]



residual_stream = embeddings + position_embeddings


x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0]
])

torch.manual_seed(0)

d_model = 4
d_k = 2
d_v = 2

W_Q = torch.randn(d_model, d_k)
W_K = torch.randn(d_model, d_k)
W_V = torch.randn(d_model, d_v)

Q = x @ W_Q
K = x @ W_K
V = x @ W_V

scores = Q @ K.T

scaled_scores = scores / math.sqrt(d_k)

attention_weights = torch.softmax(scaled_scores, dim=-1)

attention_output = attention_weights @ V

print(attention_output)


def attention_head(X, W_Q, W_K, W_V):
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.T
    scaled_scores = scores / math.sqrt(Q.shape[-1])

    attention_weights = torch.softmax(scaled_scores, dim=-1)
    output = attention_weights @ V

    return output, attention_weights

torch.manual_seed(0)

d_model = 4
num_heads = 2
head_dim = 2

# Head 1 parameters
W_Q1 = torch.randn(d_model, head_dim)
W_K1 = torch.randn(d_model, head_dim)
W_V1 = torch.randn(d_model, head_dim)

# Head 2 parameters
W_Q2 = torch.randn(d_model, head_dim)
W_K2 = torch.randn(d_model, head_dim)
W_V2 = torch.randn(d_model, head_dim)

print("W_Q1 shape:", W_Q1.shape)
print("W_Q2 shape:", W_Q2.shape)

head1_output, head1_weights = attention_head(x, W_Q1, W_K1, W_V1)
head2_output, head2_weights = attention_head(x, W_Q2, W_K2, W_V2)

print("Head 1 output shape:", head1_output.shape)
print("Head 2 output shape:", head2_output.shape)

print("\nHead 1 attention weights:")
print(head1_weights)

print("\nHead 2 attention weights:")
print(head2_weights)

multi_head_output = torch.cat([head1_output, head2_output], dim=-1)
print("Multi-head output:", multi_head_output)