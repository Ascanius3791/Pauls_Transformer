from tokenizer import encode, decode, vocab_size, vocab,data,text ,my_device
import torch


n= int(0.9*len(data))
training_data = data[:n]
testing_data = data[n:]

context_lenght = 80#16#was 10 for long training

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
    device = my_device
    x = x.to(device)
    y = y.to(device)
    return x,y

def contextify(message):
    #contextitfy the context_example
    message = torch.tensor(encode(message), dtype=torch.long)
    message = message.to(my_device)
    context_example = torch.stack([message for i in range(batch_size)])
    
    return context_example


num_loss_avg = 10*2
batch_size = 50
@torch.no_grad()
def estimate_loss(model):
    output = {}
    model.eval()
    for mode in ["training", "testing"]:
        losses = torch.zeros(num_loss_avg)
        for i in range(num_loss_avg):
            x,y = batchify(mode, batch_size)
            logits, loss = model(x,y)
            losses[i] = loss.item()
        output[mode] = losses.mean()
    model.train()
    return output
