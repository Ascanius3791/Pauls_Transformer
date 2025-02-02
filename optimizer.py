from model import BigramModel
from tokenizer import encode, decode, vocab_size, vocab,data,text, itos, stoi
from trainer import batchify, context_lenght, training_data, testing_data, estimate_loss
import torch
import torch.nn as nn
from torch.nn import functional as F

nhidden = vocab_size
nheads  = 2
nlayers = 2
batch_size = 32
embed_size=vocab_size
#make sure embed_size is multiple of nheads
embed_size = int(embed_size /nheads) * nheads#+nheads
dropout_rate=0.0

m = BigramModel(nhidden, nheads, nlayers, vocab_size, embed_size, dropout_rate)



training_iterations = 100

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("Warning: no CUDA device found. Using CPU ...")
    input("Press Enter to continue...")
optimizer = torch.optim.Adam(m.parameters(), lr = 10e-3)

m = m.to(device)

def train(model, data, optimizer, epochs):
    current_progress = 0
    for epoch in range(epochs):
        progress =int( epoch / epochs*100)
        if progress > current_progress:
            current_progress = progress
            print("Epoch: ", epoch)
            losses = estimate_loss(model)
            print("Training loss: ", losses["training"])
            print("Testing loss: ", losses["testing"])
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            print(decode(model.generate(context, max_new_tokens=50)[0].tolist()))
        xb, yb = batchify("training", batch_size=16)
        xb = xb.to(device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model

trained_model = train(m, training_data, optimizer, training_iterations)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("Model saved to", path)
    return

def restore_model(path):
    model = BigramModel(nhidden, nheads, nlayers, vocab_size, embed_size, dropout_rate)
    model.load_state_dict(torch.load(path))
    return model

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(trained_model.generate(context, max_new_tokens=50)[0].tolist()))
        

#parmaeters
