from Transformer_block import transformer_block
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=16, num_heads=2, hidden_dim=32, max_seq_len=16):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len

        # Stage 1: token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Stage 3: Q, K, V projections for all heads at once
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Stage 4: LayerNorms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Stage 4: FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

        # Stage 5: final language-model head
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape

        # -----------------------------
        # 1. Token + position embeddings
        # -----------------------------
        positions = torch.arange(seq_len, device=token_ids.device)

        tok_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)

        X = tok_emb + pos_emb

        # -----------------------------
        # 2. Transformer block: attention
        # -----------------------------
        X_norm = self.ln1(X)

        Q = self.W_Q(X_norm)
        K = self.W_K(X_norm)
        V = self.W_V(X_norm)

        # Split d_model into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Move heads before seq_len
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=token_ids.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        attention_output = attention_weights @ V

        # Merge heads back together
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.contiguous().view(batch_size, seq_len, self.d_model)

        attention_update = self.W_O(attention_output)

        # Residual add
        X = X + attention_update

        # -----------------------------
        # 3. Transformer block: FFN
        # -----------------------------
        X_norm = self.ln2(X)

        ffn_update = self.ffn(X_norm)

        # Residual add
        X = X + ffn_update

        # -----------------------------
        # 4. Vocabulary logits
        # -----------------------------
        X = self.ln_final(X)
        logits = self.lm_head(X)

        return logits