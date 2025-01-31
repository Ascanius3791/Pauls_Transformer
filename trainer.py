from tokenizer import encode, decode, vocab_size, vocab,data,text
import torch


n= int(0.9*len(data))
training_data = data[:n]
testing_data = data[n:]

context_lenght = 8

#x = training_data[:context_lenght]
#y = training_data[1:context_lenght+1]

'''
for s in range(context_lenght):
    context = x[:s+1]
    target = y[s]
    ##print("context:", decode(context), "target:", decode([target]))
    #raw
    print("context:", context, "target:", target)
'''   
def batchify(mode = "training", batch_size  =4):
    data = training_data if mode == "training" else testing_data
    ix= torch.randint(len(data)-context_lenght, (batch_size,))
    x = torch.stack([data[i:i+context_lenght] for i in ix])
    y = torch.stack([data[i+1:i+context_lenght+1] for i in ix])
    return x,y

x,y = batchify()
