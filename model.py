from tokenizer import encode, decode, vocab_size, vocab,data,text, itos, stoi
from trainer import batchify, context_lenght, training_data, testing_data
import torch
import torch.nn as nn
from torch.nn import functional as F
from Attention import Encoder, Decoder, MultiHeadedAttention, FeedForward

class BigramModel(nn.Module):
    def __init__(self, nhidden, nheads, nlayers, vocab_size, embed_size=None, dropout_rate=0.0):
        super(BigramModel, self).__init__()
        if embed_size is None:
            embed_size = vocab_size
        #print("Embeding size: ", embed_size)
        #print("Nheads: ", nheads)
        self.embedding_table = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(nhidden ,nheads ,nlayers,embed_size ,dropout_rate)
        self.decoder = Decoder(nhidden ,nheads ,nlayers,embed_size ,dropout_rate)
        self.output_layer = nn.Linear(nhidden, vocab_size)

        
    def forward(self, idx, targets = None):#if targets are provided, loss is calculated, else loss is None
        
        inputs = self.embedding_table(idx)
        
        enc_output = self.encoder(inputs)
        
        dec_output = self.decoder(inputs,enc_output)
        
        logits = self.output_layer(dec_output)
        if targets is not None:
            target_inputs = self.embedding_table(targets)
            target_enc_coutput = self.encoder(target_inputs)
            dec_output = self.decoder(target_inputs, target_enc_coutput)

        if targets is None:
            loss = None
        if targets is not None:
            # Flatten the logits and targets for cross-entropy loss
            logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * sequence_length, vocab_size)
            targets_flat = targets.view(-1)  # (batch_size * sequence_length)
            
            # Compute cross-entropy loss
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(idx) #inheritet method
            logits = logits[:, -1, :]
            likelihoods = F.softmax(logits, dim = -1)
            next_token = torch.multinomial(likelihoods, num_samples = 1)
            idx = torch.cat((idx, next_token), dim = -1) #this is now the one token longer sequence
        return idx
    

m = BigramModel(vocab_size,9,3,vocab_size,vocab_size)
#from trainer import x,y
#logits, loss = m(x,)
#input("Press Enter to continue...")
#print(logits.shape)


#print(loss)


    