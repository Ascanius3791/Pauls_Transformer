from tokenizer import encode, decode, vocab_size, vocab,data,text, itos, stoi,my_device
from trainer import batchify, context_lenght, training_data, testing_data, batch_size, estimate_loss
import torch
import torch.nn as nn
from torch.nn import functional as F
from Attention import Encoder, Decoder, MultiHeadedAttention, FeedForward, Transformer

class BigramModel(nn.Module):
    def __init__(self, nhidden, nheads, nlayers, vocab_size, embed_size=None, dropout_rate=0.0):
        super(BigramModel, self).__init__()
        if embed_size is None:
            embed_size = vocab_size
        #print("Embeding size: ", embed_size)
        #print("Nheads: ", nheads)
        self.position_embedding_table = nn.Embedding(context_lenght, embed_size)
        self.embedding_table = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(nhidden ,nheads ,nlayers,embed_size ,dropout_rate)
        self.decoder = Decoder(nhidden ,nheads ,nlayers,embed_size ,dropout_rate)
        
        self.Transformer = Transformer(nhidden,nheads,nlayers,embed_size,dropout_rate)
        #elf.output_layer = nn.Linear(nhidden, vocab_size)
        self.ln_f = nn.LayerNorm(nhidden)
        self.lm_head = nn.Linear(nhidden, vocab_size)
        self.device = my_device

        
    def forward(self, idx, targets = None,weight=None):#if targets are provided, loss is calculated, else loss is None
        B, T = idx.shape
        token_emb = self.embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        
        inputs = token_emb + pos_emb
        X = self.Transformer(inputs,mask=True,scalarproduct='standard')
        #enc_output = self.encoder(inputs,ALIBI=True,mask=True,scalarproduct='standard')

        #dec_output = self.decoder(inputs,enc_output,ALIBI=None,mask=True,scalarproduct='standard')
        
        #=dec_output
        X = self.ln_f(X)
        logits = self.lm_head(X)
    
        if targets is None:
            loss = None
        
        loss = 0
        
        if targets is not None and 1==1:#from the transformer example
            B, T, C = logits.shape
            #print("Logits shape: ", logits.shape)
            #print("logits[0][-1]: ", logits[0][-1])
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #logits = F.softmax(logits, dim = -1) ###beware this might be horribly wrong
            #logits = logits/logits.sum(-1, keepdim = True)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            #print("Logits shape: ", logits.shape)
            #print("Targets shape: ", targets.shape)
            
           # print("logits[0][-1]: ", logits[0][-1])
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, weight=weight)
            #input("press enter and here is the loss: "+str(loss))
            #print("type of loss: ", type(loss))
            
        
       # print("Exiting forward: Logits shape: ", logits.shape, "Loss: ", loss)
        
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens,mode="multinomial"):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_lenght:]
                       
            logits, loss = self.forward(idx_cond) #inheritet method
            #print("logits_bevore shape: ", logits.shape)
            #print("logits_bevore[0]: ", logits[0])
            logits = logits[:, -1, :]
            #print("logits shape: ", logits.shape)
            #print("logits[0]: ", logits[0])
            #print("logits: ", logits)
            likelyhoods = F.softmax(logits, dim = -1)
            next_token = 0
            if mode == "multinomial":
                next_token = torch.multinomial(likelyhoods, num_samples = 1)
            elif mode == "hypernomial":
                likelyhoods = likelyhoods.log()
                likelyhoods = likelyhoods/likelyhoods.sum()
                next_token = torch.multinomial(likelyhoods, num_samples = 1)
            elif mode == "greedy":
                next_token = torch.argmax(likelyhoods, dim = -1, keepdim = True)
            else:
                raise ValueError("Mode must be either 'multinomial' or 'greedy'")
            idx = torch.cat((idx, next_token), dim = -1) #this is now the one token longer sequence
        return idx
    

m = BigramModel(vocab_size,9,3,vocab_size,vocab_size)
#from trainer import x,y
#logits, loss = m(x,)
#input("Press Enter to continue...")
#print(logits.shape)


#print(loss)


    