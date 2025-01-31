import torch

#base_data = "short_Goethe.txt" # for testing
base_data = "Goethe.txt" # for final run
with open(base_data, "r") as file:
    char_to_remove = ["*"]
    text = file.read()
    #text = text.lower()
    for char in char_to_remove:
        text = text.replace(char, "")
    text = text.replace("  ", " ")
#print(text[:1000])

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
#print("Vocabulary size:", vocab_size)
print(vocab)


stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

def encode(text):
    return [stoi[ch] for ch in text]

def decode(tokens):
    return ''.join([itos[token] for token in tokens])


#print(encode("hallo"))
#print(decode([1,2,3,3,4]))
#print(decode(encode("hallo")))

data = torch.tensor(encode(text), dtype=torch.long)