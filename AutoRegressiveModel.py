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

token_to_id ={token:idx for idx,token in enumerate(vocab)}
id_to_token = {idx:token for idx,token in enumerate(vocab)}

