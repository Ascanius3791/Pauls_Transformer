# in this python file the attention layer is implemeted
import torch 
import torch.nn as nn
from torch.nn import functional as F


#two perceptron layer for decoder take it batchwise
class FeedForward(nn.Module):
    def __init__(self,nhidden,inpsize,dropout_rate=0.0):
        super(FeedForward,self).__init__()
        # create two layers of the feed forward neural network used in the encoder 
        self.Layer1=nn.Linear(nhidden,4*inpsize)#4 is arnes tip
        #use ReLU as activation function
        self.relu=nn.ReLU()
        self.Layer2=nn.Linear(4*inpsize,nhidden)

        self.dropout=nn.Dropout(dropout_rate)
   
    #define forward function of two layers
    def forward(self,x):
        # first layer of neural network 
        x=self.Layer1(x)
        x=self.relu(x)
        x=self.dropout(x)
        #second layer of neural network 
        x=self.Layer2(x)

        return x 
    
    def weights_biases(self):
        
        weights1=self.Layer1.weight
        weights2=self.Layer2.weight
        bias1=self.Layer1.bias
        bias2=self.Layer2.bias 

        return weights1,weights2,bias1,bias2
   
class MultiHeadedAttention(nn.Module):
    # define constructor were query and weights are initialized 
    def __init__ (self,nhidden,nheads=10,dropout_rate=0.0):
        super(MultiHeadedAttention,self).__init__()
        self.nhidden=nhidden
        self.nheads=nheads 
        self.query=nn.Linear(nhidden,nhidden,bias=False)
        self.key=nn.Linear(nhidden,nhidden,bias=False)
        self.value=nn.Linear(nhidden,nhidden,bias=False)
        self.wo=nn.Linear(int(nhidden/nheads),nhidden,bias=False)
        self.linear_bias = nn.Parameter(torch.arange(0, nheads).float().unsqueeze(1))
        self.dropout=nn.Dropout(dropout_rate)
        #self.MHA=nn.MultiheadAttention(nhidden,nheads,dropout_rate,bb)

        # initialize weights 
     # x has form xbatch, size, token
    def forward(self,x,ALIBI=False,mask=None,scalarproduct='standard',q=None):
        B, T, C=x.size()
        Q=self.query(x)
        if q is not None:
            Q=self.query(q)

        K=self.key(x)
        V=self.value(x)
        
        sub_token=C/self.nheads
        if sub_token!=int(sub_token):
            raise ValueError('sub_token must be integer')
        sub_token=int(sub_token)#change float to int
        #print('sub_token',sub_token)
        
        Q=Q.reshape(B,T,self.nheads,sub_token).permute(0,2,1,3)#changed C to sub token, such dat the dimensions line up for reshape
        K=K.reshape(B,T,self.nheads,sub_token).permute(0,2,1,3)
        V=V.reshape(B,T,self.nheads,sub_token).permute(0,2,1,3)
        

        #define attention weights 

        #normalize product 

        #transpose second last and last dimension of Key K 

        dk=Q.size()[-1] #last dimension 

        prod=torch.matmul(Q,K.transpose(-2,-1))/torch.math.sqrt(dk)
        #print('prod',prod.size())
        #input('prod')

        #define causal mask for product 

        #casual mask 

        
        


        if ALIBI==True:
            offset = [[[[(i-j)*2**(-8*float(h+1)/self.nheads) for i in range(T)] for j in range(T)] for h in range(self.nheads)]for _ in range(B)]
            offset=torch.tensor(offset,dtype=torch.float)
            offset=offset.to(device=x.device, dtype=x.dtype)
            prod+=offset
            
        if mask is not None:
            mask = torch.tril(torch.ones(T, T))
            mask = mask.to(device=x.device, dtype=x.dtype)
             
            prod=prod.masked_fill(mask == 0,float('-inf'))
        else:
            raise ValueError('mask must be provided')        
        attention_weights=F.softmax(prod,2)
        
        
        
        attention_weights=self.dropout(attention_weights)
        
        attention_output=torch.matmul(attention_weights,V)
    
        #print('attention_output',attention_output.size())
        outputs=self.wo(attention_output)
        outputs=outputs.sum(1)
        #print('outputs',outputs.size())
        #input('outputs')
        x=x+outputs

        
           
        return x 

class Block(nn.Module):
    def __init__(self,nhidden,inpsize,nheads=10,dropout_rate=0.0):
        super(Block,self).__init__()

        #define attention norm 

        self.attention_norm=nn.LayerNorm(nhidden,eps=1e-6)
        
        self.attention=MultiHeadedAttention(nhidden,nheads,dropout_rate)
        

        self.attention_dropout=nn.Dropout(dropout_rate)

        self.network_norm=nn.LayerNorm(nhidden,eps=1e-6)
        self.network=FeedForward(nhidden,inpsize,dropout_rate)
    
    # define forward function of encoder layer 

    def forward(self,x,ALIBI=False,mask=None,scalarproduct='standard'):
        
        y=self.attention_norm(x)
        
        y=self.attention(y,ALIBI,mask,scalarproduct)
        y=x+self.attention_dropout(y)

        

        z=self.network_norm(y)
        z=y+self.network(z)

        return z


# define one layer of encoding with attention layer+ feed forward neural network 
class EncoderLayer(nn.Module):
    def __init__(self,nhidden,inpsize,nheads=10,dropout_rate=0.0):
        super(EncoderLayer,self).__init__()

        #define attention norm 

        self.attention_norm=nn.LayerNorm(nhidden,eps=1e-6)
        
        self.attention=MultiHeadedAttention(nhidden,nheads,dropout_rate)
        

        self.attention_dropout=nn.Dropout(dropout_rate)

        self.network_norm=nn.LayerNorm(nhidden,eps=1e-6)
        self.network=FeedForward(nhidden,inpsize,dropout_rate)
    
    # define forward function of encoder layer 

    def forward(self,x,ALIBI=False,mask=None,scalarproduct='standard'):
        
        y=self.attention_norm(x)
        
        y=self.attention(y,ALIBI,mask,scalarproduct)
        y=self.attention_dropout(y)

        

        y=self.network_norm(y)
        y=self.network(y)

        return y
   
#define one lyer of Decoding 

class DecoderLayer(nn.Module):
     def __init__(self,nhidden,inpsize,nheads=10,dropout_rate=0.0):
        super(DecoderLayer,self).__init__()

        #define attention norm 

        self.attention_norm=nn.LayerNorm(nhidden,eps=1e-6)

        self.attention=MultiHeadedAttention(nhidden,nheads,dropout_rate)
        self.enc_dec_attention=MultiHeadedAttention(nhidden,nheads,dropout_rate)
        self.enc_dec_attention_norm=nn.LayerNorm(nhidden,eps=1e-6)

        self.dropout=nn.Dropout(dropout_rate)

        self.network_norm=nn.LayerNorm(nhidden,eps=1e-6)
        self.network=FeedForward(nhidden,inpsize,dropout_rate)
    
    # define forward function of encoder layer 

     def forward(self,x,enc_output,ALIBI=False,mask=None,decmask=None,scalarproduct='standard'):
        
        tmp=self.attention_norm(x)
        # this is the decoder attention output 
        tmp=self.attention(tmp,ALIBI,mask,scalarproduct)


        #define encoder decoder mechanism 

        if enc_output is not None:
            tmp2=self.enc_dec_attention_norm(tmp)
            tmp3=self.enc_dec_attention(enc_output,ALIBI,mask,scalarproduct,tmp2)


        out=self.network_norm(tmp3)
        out=self.network(out)


        return out 
   
#define encoder and decoder classes 

class Encoder(nn.Module):
    def __init__(self,nhidden,nheads,nlayers,inpsize,dropout_rate=0.0):
         super(Encoder,self).__init__()

         # define array of encoder layers 
        # print('nheads_Enc',nheads)

         encoders=[EncoderLayer(nhidden,inpsize,nheads,dropout_rate) for _ in range(nlayers)]

         self.layers=nn.ModuleList(encoders)

         #output of last layer is normalized 

         self.last_norm=nn.LayerNorm(nhidden,eps=1e-6)
         self.shape_matcher=nn.Linear(inpsize,nhidden)
    
    def forward(self,inputs,ALIBI=False,mask=None,scalarproduct='standard'):
        
        outputs=self.shape_matcher(inputs)
        

        for layer in self.layers:
            outputs=layer(outputs,ALIBI,mask,scalarproduct)
        
        outputs=self.last_norm(outputs)
        return outputs

class Decoder(nn.Module):

     def __init__(self,nhidden,nheads,nlayers,inpsize,dropout_rate=0.0):
            super(Decoder,self).__init__()#change Dencoder to Decoder

            # define array of encoder layers 
            #print('nheadsDec',nheads)
            decoders=[DecoderLayer(nhidden,inpsize,nheads,dropout_rate) for _ in range(nlayers)]

            self.layers=nn.ModuleList(decoders)

            #output of last layer is normalized 

            self.last_norm=nn.LayerNorm(nhidden,eps=1e-6)
            self.shape_matcher=nn.Linear(inpsize,nhidden)
     
     def forward(self,targets,enc_output,ALIBI=False,mask=None,decmask=None,scalarproduct='standard'):
        outputs=self.shape_matcher(targets)

        for _, dec_layer in enumerate(self.layers):
            outputs=dec_layer(outputs,enc_output,ALIBI,mask,scalarproduct)
        
        outputs=self.last_norm(outputs)
        return outputs

class Transformer(nn.Module):
    def __init__(self,nhidden,nheads,nlayers,inpsize,dropout_rate=0.0):
         super(Transformer,self).__init__()

         # define array of encoder layers 
        # print('nheads_Enc',nheads)

         Blocks=[Block(nhidden,inpsize,nheads,dropout_rate) for _ in range(nlayers)]

         self.layers=nn.ModuleList(Blocks)

         #output of last layer is normalized 

         self.last_norm=nn.LayerNorm(nhidden,eps=1e-6)
         self.shape_matcher=nn.Linear(inpsize,nhidden)
    
    def forward(self,inputs,ALIBI=False,mask=None,scalarproduct='standard'):
        
        outputs=self.shape_matcher(inputs)
        

        for layer in self.layers:
            outputs=layer(outputs,ALIBI,mask,scalarproduct)
        
        outputs=self.last_norm(outputs)
        return outputs
 

