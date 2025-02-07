from model import BigramModel
from tokenizer import encode, decode, vocab_size, vocab,data,text, itos, stoi,my_device,weight
from trainer import batchify, context_lenght, training_data, testing_data, estimate_loss, batch_size, contextify
import torch
import torch.nn as nn
from torch.nn import functional as F

nhidden = 8#20   #always used 6,3,2
nheads  = 4#5
nlayers = 8
#batch_size =32 go to trainer.py for changes
embed_size=vocab_size# jakobs tip: 200+
#make sure embed_size is multiple of nheads
#embed_size = int(embed_size /nheads) * nheads+nheads

dropout_rate=0.1

m = BigramModel(nhidden, nheads, nlayers, vocab_size, embed_size, dropout_rate)

def save_model(model, path):
    print("Saving model to", path)
    torch.save(model.state_dict(), path)
    print("Model saved to", path)
    return

def restore_model(path):
    model = BigramModel(nhidden, nheads, nlayers, vocab_size, embed_size, dropout_rate)
    model.load_state_dict(torch.load(path))
    return model

#check, if there is already a trained model at "trained_model.pth"
try:
    m.load_state_dict(torch.load("trained_model.pth"))
    print("Model restored from trained_model.pth")
except:
    print("No trained model found. Training a new model ...")


training_iterations = 10*5000

if my_device == torch.device("cuda"):
    pass
else:
    print("Warning: no CUDA device found. Using CPU ...")
    input("Press Enter to continue...")
optimizer = torch.optim.Adam(m.parameters(), lr = 10e-3)

m = m.to(my_device)

def train(model, data, optimizer, epochs,weight=None):
    #current_progress = 0
    for epoch in range(epochs):
        
        if epoch % 25 == 0:
            save_model(model, "trained_model.pth")
            #current_progress = progress
            print("Epoch: ", epoch)
            losses = estimate_loss(model)
            print("Training loss: ", losses["training"])
            print("Testing loss: ", losses["testing"])
            context = torch.zeros((1, 1), dtype=torch.long, device=my_device)
            #define alternative context, "Des Pudels Kern"
            context = contextify("des pudels kern ist")
            context = context.to(my_device)
            print(decode(model.generate(context, max_new_tokens=50,mode="greedy")[0].tolist()))
        xb, yb = batchify("training", batch_size=batch_size)
        xb = xb.to(my_device)
        logits, loss = model(xb, yb,weight=weight)
        if epoch % 100 == 0 or epoch < 10:
            temp_xb, temp_yb = batchify("training", batch_size=1)
            temp_xb = temp_xb.to(my_device)
            temp_logits, temp_loss = model(temp_xb, temp_yb,weight=weight)
            context = temp_xb[0].tolist()
            solution = temp_yb[0].tolist()
            solution = decode(solution)[-1]
            #print("Batches: ", len(temp_xb))
            print("Context: ", decode(context))
            if solution == " ": solution = "SPACE"
            if solution == "\n": solution = "NEWLINE"
            print("Solution: ", solution)
            
            print("Predictions: ")
            likelihoods = F.softmax(temp_logits[0], dim=-1)
            for i, p in enumerate(likelihoods):
                if(p.item() > 0.001):
                    print(decode([i]), ":", p.item())
            print("Loss of this batch: ", temp_loss.item())
            
        '''
        print("\nTraining Step: ", epoch)
        print("xb[0] : ", xb[0])
        print("yb[0] : ", yb[0])
        #print("Logits shape: ", logits.shape)
        print("Logits[0]: ", logits[0])
        print("Loss: ", loss)
        input("Loss calculated\n")
        '''
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model

trained_model = train(m, training_data, optimizer, training_iterations,weight=weight)



context = torch.zeros((1, 1), dtype=torch.long, device=my_device)
print(decode(trained_model.generate(context, max_new_tokens=50)[0].tolist()))

save_model(trained_model, "trained_model.pth")
        

#parmaeters
