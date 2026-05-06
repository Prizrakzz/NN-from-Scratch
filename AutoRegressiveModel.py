from Transformer_block import transformer_block
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

sentences = [
    "i like cats",
    "i like dogs",
    "you like cats",
    "cats eat fish"
]

sentences = [s + " <END>" for s in sentences]
tokenized = [s.split() for s in sentences]

all_tokens = []
for sentence in tokenized:
    for token in sentence:
        all_tokens.append(token)

vocab = sorted(set(all_tokens))

#lookup tables
token_to_id ={token:idx for idx,token in enumerate(vocab)}
id_to_token = {idx:token for idx,token in enumerate(vocab)}


encoded_sentences = []
for sentence in tokenized:
    ids = [token_to_id[token] for token in sentence]
    encoded_sentences.append(ids)

print(encoded_sentences)

inputs = []
targets = []

for ids in encoded_sentences:
    x = ids[:-1]
    y = ids[1:]

    inputs.append(x)
    targets.append(y)

X_train = torch.tensor(inputs)
Y_train = torch.tensor(targets)

print("X_train:")
print(X_train)

print("\nY_train:")
print(Y_train)

print("\nX_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

