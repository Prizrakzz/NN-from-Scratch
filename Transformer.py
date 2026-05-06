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
