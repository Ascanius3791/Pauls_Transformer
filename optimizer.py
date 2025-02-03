from model import BigramModel
from tokenizer import encode, decode, vocab_size, vocab,data,text, itos, stoi
from trainer import batchify, context_lenght, training_data, testing_data
import torch
import torch.nn as nn
from torch.nn import functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("Warning: no CUDA device found. Using CPU ...")
    input("Press Enter to continue...")
optimizer = torch.optim.Adam(m.parameters(), lr = 10e-3)

#parmaeters
