sentences = [
    "i like cats",
    "i like dogs",
    "you like cats",
    "cats eat fish"
]

tokenized_sentences = [sentence.split() for sentence in sentences]

print(tokenized_sentences)

all_tokens = []

for sentence in tokenized_sentences:
    for token in sentence:
        all_tokens.append(token)

vocab = sorted(set(all_tokens))

token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}


print("vocabulary:", vocab)
print("Token to ID:", token_to_id)
print("ID to token:", id_to_token)


sentencetwo = "i like cats"

tokens = sentencetwo.split()
token_ids = [token_to_id[token] for token in tokens]

print("Tokens:", tokens)
print("Token IDs:", token_ids)
