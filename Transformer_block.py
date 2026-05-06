import torch
import math

# -----------------------------
# 1. Tiny residual stream
# -----------------------------
X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],  # token 0: "i"
    [0.0, 1.0, 1.0, 0.0],  # token 1: "like"
    [0.0, 1.0, 0.0, 1.0],  # token 2: "cats"
])

print("Initial X shape:", X.shape)

# -----------------------------
# 2. LayerNorm
# -----------------------------
def layer_norm_with_params(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    return gamma * normalized + beta

# -----------------------------
# 3. One causal attention head
# -----------------------------
def causal_attention_head(X, W_Q, W_K, W_V):
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.T
    scaled_scores = scores / math.sqrt(Q.shape[-1])

    seq_len = X.shape[0]
    mask = torch.tril(torch.ones(seq_len, seq_len))

    masked_scores = scaled_scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = torch.softmax(masked_scores, dim=-1)

    output = attention_weights @ V

    return output, attention_weights

# -----------------------------
# 4. Multi-head causal attention
# -----------------------------
def multi_head_causal_attention(X, params):
    head_outputs = []
    attention_weights_all = []

    for head in params["heads"]:
        head_output, attention_weights = causal_attention_head(
            X,
            head["W_Q"],
            head["W_K"],
            head["W_V"]
        )

        head_outputs.append(head_output)
        attention_weights_all.append(attention_weights)

    concatenated = torch.cat(head_outputs, dim=-1)
    projected = concatenated @ params["W_O"]

    return projected, attention_weights_all

# -----------------------------
# 5. Feed-forward network
# -----------------------------
def feed_forward(X, params):
    hidden = X @ params["W1"] + params["b1"]
    activated = torch.relu(hidden)
    output = activated @ params["W2"] + params["b2"]
    return output

# -----------------------------
# 6. Transformer block
# -----------------------------
def transformer_block(X, params):
    print("\nInput to block:", X.shape)

    # LayerNorm before attention
    X_norm_1 = layer_norm_with_params(
        X,
        params["ln1_gamma"],
        params["ln1_beta"]
    )
    print("After LayerNorm 1:", X_norm_1.shape)

    # Multi-head causal attention
    attention_update, attention_weights_all = multi_head_causal_attention(
        X_norm_1,
        params
    )
    print("Attention update:", attention_update.shape)

    # Residual connection
    X = X + attention_update
    print("After attention residual add:", X.shape)

    # LayerNorm before FFN
    X_norm_2 = layer_norm_with_params(
        X,
        params["ln2_gamma"],
        params["ln2_beta"]
    )
    print("After LayerNorm 2:", X_norm_2.shape)

    # FFN
    ffn_update = feed_forward(X_norm_2, params)
    print("FFN update:", ffn_update.shape)

    # Residual connection
    X = X + ffn_update
    print("After FFN residual add:", X.shape)

    return X, attention_weights_all

# -----------------------------
# 7. Create tiny parameters
# -----------------------------
torch.manual_seed(0)

d_model = 4
num_heads = 2
head_dim = d_model // num_heads
hidden_dim = 8

params = {
    "heads": [],
    "W_O": torch.randn(d_model, d_model),

    "ln1_gamma": torch.ones(d_model),
    "ln1_beta": torch.zeros(d_model),

    "ln2_gamma": torch.ones(d_model),
    "ln2_beta": torch.zeros(d_model),

    "W1": torch.randn(d_model, hidden_dim),
    "b1": torch.zeros(hidden_dim),

    "W2": torch.randn(hidden_dim, d_model),
    "b2": torch.zeros(d_model),
}

for _ in range(num_heads):
    head = {
        "W_Q": torch.randn(d_model, head_dim),
        "W_K": torch.randn(d_model, head_dim),
        "W_V": torch.randn(d_model, head_dim),
    }
    params["heads"].append(head)

# -----------------------------
# 8. Run one Transformer block
# -----------------------------
output, attention_weights_all = transformer_block(X, params)

print("\nFinal output:")
print(output)

print("\nFinal output shape:", output.shape)

print("\nAttention weights from head 1:")
print(attention_weights_all[0])

print("\nAttention weights from head 2:")
print(attention_weights_all[1])