import torch

my_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#my_device = torch.device("cpu")
torch.set_num_threads(4)
#base_data = "short_Goethe.txt" # for testing
base_data = "Goethe.txt" # for final run
#base_data = "super_short_Goethe.txt" # for testing

with open(base_data, "r") as file:
    char_to_remove = ["*","\!","?",".",",",";",":","-","_","(",")","[","]","{","}","<",">","|","/","\\","\"","'","`","^","~","@","#","$","%","&","=","+","*","§","°","1","2","3","4","5","6","7","8","9","0"]
    text = file.read()
    text = text.lower()
    for char in char_to_remove:
        text = text.replace(char, " ")
    text = text.replace("  ", " ")
    #map all special characters to " "
text = text.replace('ä', 'ae')
text = text.replace('ö', 'oe')
text = text.replace('ü', 'ue')
text = text.replace('ß', 'ss')    
#print(text[:1000])

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
#remove weird characters(after z)
redacted_vocab = [c for c in vocab if ord(c) < 123 and c not in sorted(['ä','ö','ü','ß'])]
diff_vocab = set(vocab) - set(redacted_vocab)
#print("diff_vocab:", diff_vocab)
#remove diff_vocab from text
#print("Characters removed:", diff_vocab)
for char in diff_vocab:
    text = text.replace(char, " ")

vocab = redacted_vocab
vocab_size = len(vocab)
#print("Vocabulary size:", vocab_size)
#print(vocab)


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
#get the appearance count of each character
char_count = []
for i in range(vocab_size):
    char_count.append(text.count(itos[i]))
print(char_count)
weight = torch.tensor(char_count, dtype=torch.long)
weight = 1/weight.sqrt()
weight = weight / weight.sum()
weight = weight.to(my_device)
#print letters and their weights
#for i in range(vocab_size):
    #print(itos[i], weight[i].item())
#input("Press Enter to continue...")
#print("Text[0:500]:", text[0:500])


#context_example = torch.tensor(encode("das ist des pudels kern!"), dtype=torch.long)
#context_example = torch.tensor(encode("das ist des pudels kern!"), dtype=torch.long)
#print("Example contest_coded:", context_example)
#print("Example context:", decode(context_example))
#print(decode(encode(context_example).tolist()))