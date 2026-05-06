import torch
import torch.nn as nn

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
print(embeddings)

max_seq_length = 3

position_embedding_table = torch.randn(max_seq_length, embedding_dim)

position_ids = torch.arange(max_seq_length)

position_embeddings = position_embedding_table[position_ids]

print(position_embeddings)

residual_stream = embeddings + position_embeddings
print(residual_stream)



