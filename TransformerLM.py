from Transformer_block import transformer_block
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------
# 1. Tiny dataset
# -----------------------------
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

token_to_id = {token: i for i, token in enumerate(vocab)}
id_to_token = {i: token for token, i in token_to_id.items()}

vocab_size = len(vocab)

print("Vocabulary:")
print(token_to_id)

# -----------------------------
# 2. Encode data
# -----------------------------
encoded_sentences = []

for sentence in tokenized:
    ids = [token_to_id[token] for token in sentence]
    encoded_sentences.append(ids)

inputs = []
targets = []

for ids in encoded_sentences:
    inputs.append(ids[:-1])
    targets.append(ids[1:])

X_train = torch.tensor(inputs)
Y_train = torch.tensor(targets)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

# -----------------------------
# 3. Transformer LM
# -----------------------------
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=16, num_heads=2, hidden_dim=32, max_seq_len=16):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len

        assert d_model % num_heads == 0

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape

        positions = torch.arange(seq_len, device=token_ids.device)

        tok_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)

        X = tok_emb + pos_emb

        # Attention sublayer
        X_norm = self.ln1(X)

        Q = self.W_Q(X_norm)
        K = self.W_K(X_norm)
        V = self.W_V(X_norm)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=token_ids.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        attention_output = attention_weights @ V

        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.contiguous().view(batch_size, seq_len, self.d_model)

        attention_update = self.W_O(attention_output)

        X = X + attention_update

        # FFN sublayer
        X_norm = self.ln2(X)
        ffn_update = self.ffn(X_norm)

        X = X + ffn_update

        # Vocabulary prediction
        X = self.ln_final(X)
        logits = self.lm_head(X)

        return logits

# -----------------------------
# 4. Train
# -----------------------------
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

# -----------------------------
# 5. Generate
# -----------------------------
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

print(generate(model, "i"))
print(generate(model, "you"))
print(generate(model, "cats"))