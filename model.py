from tokenizer import encode, decode, vocab_size, vocab,data,text, itos, stoi
from trainer import batchify, context_lenght, training_data, testing_data
import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramModel(nn.Module):
    def __init__(self, vocab_size,embed_size=None):
        super(BigramModel, self).__init__()
        if embed_size is None:
            embed_size = vocab_size
        self.embedding_table = nn.Embedding(vocab_size, embed_size)
        
        
    def forward(self, idx, targets = None):
        logits = self.embedding_table(idx)
        print("logits",logits)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape #batch, token, vector 
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(idx) #inheritet method
            logits = logits[:, -1, :]
            likelihoods = F.softmax(logits, dim = -1)
            next_token = torch.multinomial(likelihoods, num_samples = 1)
            idx = torch.cat((idx, next_token), dim = -1) #this is now the one token longer sequence
        return idx
    

m = BigramModel(vocab_size)
from trainer import x,y
logits, loss = m(x, y)
print(logits.shape)


print(loss)
generated = m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)
print(generated.shape)
print(decode(generated[0].tolist()))

    