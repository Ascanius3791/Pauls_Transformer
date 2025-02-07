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


#check, if there is already a trained model at "trained_model.pth"
try:
    m.load_state_dict(torch.load("trained_model.pth"))
    print("Model restored from trained_model.pth")
except:
    print("No trained model found. Training a new model ...")


m = m.to(my_device)
def generator(model, use_data = None,mode = "greedy",max_new_tokens=50):
    #current_progress = 0
    while True:
        
        context = batchify(mode="testing", batch_size=1)[0]
        #define alternative context, "Des Pudels Kern"
        if use_data is None:
            print("Enter Context:")
            input_string = "INVALID"
            while input_string != input_string.lower() or len(input_string) < 1:
                input_string = input("Enter Context: ")
            context = contextify(input_string).to(my_device)
            print("Result:")
            print(decode(model.generate(context, max_new_tokens=max_new_tokens,mode="greedy")[0].tolist()))
        else:
            print("Context:")
            print(decode(context[0].tolist()))
            print("Result:")
            print(decode(model.generate(context, max_new_tokens=max_new_tokens,mode="greedy")[0].tolist()))

generator(m, None,mode = "hypernomial",max_new_tokens=500)

#parmaeters
