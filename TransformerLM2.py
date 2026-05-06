import math as m
import torch 
import torch.nn as nn
import torch.nn.functional as F

sentences = [
    "a person who thinks about the future",
    "that is not the way to go",
    "a person is the way he thinks",
    "i hate that person and his vision",
    "the future is not the way forward",
    "he thinks about the person he hates",
    "a visionary person thinks about the future",
    "that person is not the way forward",
]

sentences = [s + " <END>" for s in sentences]

tokenized_sentences = [s.split() for s in sentences]

alltokens = []

for s in tokenized_sentences:
    for token in s:
        alltokens.append(token)

vocab = sorted(set(alltokens))

token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}

vocab_size = len(vocab)

encoded_sentences = []

for sentence in tokenized_sentences:
    ids = [token_to_id[token] for token in sentence]
    encoded_sentences.append(ids)

inputs = []
targets = []

for ids in encoded_sentences:
    x = ids[:-1]
    y = ids[1:]

    inputs.append(x)
    targets.append(y)

X_train = torch.tensor(inputs)
Y_train = torch.tensor(targets)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_module = 16, num_heads = 2, hidden_dim = 32, max_seq_length = 16):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_module = d_module
        self.num_heads = num_heads
        self.head_dim = d_module // num_heads
        self.max_seq_length = max_seq_length
        assert d_module % num_heads == 0

        self.token_embedding = nn.Embedding(vocab_size, d_module)
        self.position_embedding = nn.Embedding(max_seq_length, d_module)


        self.W_Q = nn.Linear(d_module, d_module, bias=False)
        self.W_K = nn.Linear(d_module, d_module, bias=False)
        self.W_V = nn.Linear(d_module, d_module, bias=False)
        self.W_O = nn.Linear(d_module, d_module, bias=False)

        self.ln1 = nn.LayerNorm(d_module)
        self.ln2 = nn.LayerNorm(d_module)

        self.ffn = nn.Sequential(
            nn.Linear(d_module, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_module)
        )

        self.ln_final = nn.LayerNorm(d_module)
        self.lm_head = nn.Linear(d_module, vocab_size, bias=False)

    def forward(self, token_ids):
        batch_size, seq_length = token_ids.shape

        positions = torch.arange(seq_length, device=token_ids.device)

        tok_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)

        X = tok_emb + pos_emb

        X_norm = self.ln1(X)

        Q = self.W_Q(X_norm)
        K = self.W_K(X_norm)
        V = self.W_V(X_norm)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)

        scores = Q @ K.transpose(-2, -1) / m.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(seq_length, seq_length, device=token_ids.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        attention_output = attention_weights @ V

        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_module)

        attention_output = self.W_O(attention_output)

        X = X + attention_output

        #FFN sublayer
        X_norm = self.ln2(X)
        ffn_update = self.ffn(X_norm)

        X = X + ffn_update

        X = self.ln_final(X)

        logits = self.lm_head(X)

        return logits
        
torch.manual_seed(0)

model = TransformerLM(vocab_size=vocab_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

for step in range(300):
    logits = model(X_train)

    logits_flat = logits.view(-1, vocab_size)
    targets_flat = Y_train.view(-1)

    loss = F.cross_entropy(logits_flat, targets_flat)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"step {step}, loss = {loss.item():.4f}")

def generate(model, start_text, max_new_tokens=5):
    model.eval()

    tokens = start_text.split()
    token_ids = [token_to_id[token] for token in tokens]

    for _ in range(max_new_tokens):
        x = torch.tensor([token_ids])

        logits = model(x)

        last_logits = logits[0, -1, :]

        next_id = torch.argmax(last_logits).item()
        next_token = id_to_token[next_id]

        token_ids.append(next_id)

        if next_token == "<END>":
            break

    return " ".join(id_to_token[i] for i in token_ids)

print(generate(model, "a person who thinks"))
print(generate(model, "the future is"))
print(generate(model, "he thinks about"))





        

