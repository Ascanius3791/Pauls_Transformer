# in this python file the attention layer is implemeted
import torch 
import torch.nn as nn
from torch.nn import functional as F


#two perceptron layer for decoder take it batchwise
class FeedForward(nn.Module):
    def __init__(self,nhidden,inpsize,dropout_rate=0.0):
        super(FeedForward,self).__init__()
        # create two layers of the feed forward neural network used in the encoder 
        self.Layer1=nn.Linear(nhidden,inpsize)
        #use ReLU as activation function
        self.relu=nn.ReLU()
        self.Layer2=nn.Linear(inpsize,nhidden)

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
        self.wo=nn.Linear(nhidden,nhidden,bias=False)
        self.linear_bias = nn.Parameter(torch.arange(0, nheads).float().unsqueeze(1))
        self.dropout=nn.Dropout(dropout_rate)

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
        #print('Q',Q.size())
        #print("X.size",x.size())
        #print('Q.shape',Q.shape)
        #Q=Q.reshape(B,T,self.nheads,sub_token).permute(0,2,3,1)#changed C to sub token, such dat the dimensions line up for reshape
        #K=K.reshape(B,T,self.nheads,sub_token).permute(0,2,3,1)
        #V=V.reshape(B,T,self.nheads,sub_token).permute(0,2,3,1)
        #print("Shape of Q",Q.size())
        #print("Attention matrices are defined")

        #define attention weights 

        #normalize product 

        #transpose second last and last dimension of Key K 

        dk=Q.size()[-1] #last dimension 

        prod=torch.matmul(Q,K.transpose(-2,-1))/torch.math.sqrt(dk)

        #define causal mask for product 

        #casual mask 

        if mask is not None:
            mask = torch.tril(torch.ones(T, T))
            mask = mask.to(device=x.device, dtype=x.dtype)
           
            #print('prod',prod.size())
            #print("mask = " , mask.size())
             
            prod=prod.masked_fill(mask == 0,float('-inf'))
            #print('prod[0]', prod[0])
            #print("softmax",F.softmax(prod,2))
            #input("prod after mask")
        


        if ALIBI==True:
         
            slope=torch.Tensor([2**(-8*(i+1)/self.nheads) for i in range(self.nheads)],dtype=int).unsqueeze(1).unsqueeze(1).unsqueeze(0)


         
            bias=slope*torch.arange(-prod.size()[-1]+1,1,1).float().unsqueeze(0).unsqueeze(0).unsqueeze(2)

            prod+=bias 
            print("bias.shape",bias.shape)
        

        #print('prod[0]', prod[0])        
        attention_weights=F.softmax(prod,2)
        #print('attention_weights',attention_weights.size())
        #print('attention_weights[0]',attention_weights[0])
        #input("attention_weights")
        
        
        attention_weights=self.dropout(attention_weights)
        
        attention_output=torch.matmul(attention_weights,V)
        #print("nhidden",self.nhidden)
        #define output 
        #print('attention_output',attention_output.size())
        #print("attention. output",attention_output.shape)
        
        outputs=self.wo(attention_output)

        
           
        return outputs 

# define one layer of encoding with attention layer+ feed forward neural network 
class EncoderLayer(nn.Module):
    def __init__(self,nhidden,inpsize,nheads=10,dropout_rate=0.0):
        super(EncoderLayer,self).__init__()

        #define attention norm 

        self.attention_norm=nn.LayerNorm(nhidden,eps=1e-6)
        #print('nhaeds',nheads)
        #print("begin attention")
        self.attention=MultiHeadedAttention(nhidden,nheads,dropout_rate)
        #print("end attention")

        self.attention_dropout=nn.Dropout(dropout_rate)

        self.network_norm=nn.LayerNorm(nhidden,eps=1e-6)
        self.network=FeedForward(nhidden,inpsize,dropout_rate)
    
    # define forward function of encoder layer 

    def forward(self,x,ALIBI=False,mask=None,scalarproduct='standard'):
        #print('x_encoder_layer',x.size())
        #print('x_encoder_layer',x.shape)
        y=self.attention_norm(x)
        #print('y_encoder_layer',y.size())
        #print('y_encoder_layer',y.shape)
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
            tmp3=self.enc_dec_attention(enc_output,False,decmask,scalarproduct,tmp2)


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
    
    def forward(self,inputs,ALIBI=False,mask=None,scalarproduct='standard'):
        outputs=inputs

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
     
     def forward(self,targets,enc_output,ALIBI=False,mask=None,decmask=None,scalarproduct='standard'):
        outputs=targets

        for _, dec_layer in enumerate(self.layers):
            outputs=dec_layer(outputs,enc_output,ALIBI,decmask,scalarproduct)
        
        outputs=self.last_norm(outputs)
        return outputs
   

