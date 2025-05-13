import torch
import torch.nn as nn
import numpy as np 
from PositionalEncoder import PositionalEncoder
from FFN import FFN

 
class EncoderBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, d_hidden, num_hidden_layers, pdrop):
        super(EncoderBlock, self).__init__()
        
        """masked mh attention -> add norm -> feedforward -> add norm"""
        self.mh_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True) # can add dropout here if we want
        
        # layer norm normalizes each embedding vector individually, so it needs the size of those vectors (d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.ffn = FFN(d_input=d_model, d_output=d_model, d_hidden=d_hidden, num_hidden_layers=num_hidden_layers) # feed forward
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x, padding_mask):
        """ forward method of a mhattention object needs q, k, v (it handles the projection of the inputs),
            and an attention mask
        """

        attn_output, att_weights = self.mh_attention(x, x, x, key_padding_mask=padding_mask)
        attn_output = self.dropout(attn_output) # apply some dropout
        residual_one = x + attn_output # first res connection
        normalized = self.layer_norm(residual_one)
        ffn_output = self.ffn(normalized)
        ffn_output = self.dropout(ffn_output) # EXPERIMENT WITH THIS
        
        out = residual_one + ffn_output # second res connection
        
        return out

# wrapper for the full stack of encoder blocks
class Encoder(nn.Module):
    def __init__(self, num_heads, num_hidden_layers, num_encoders, d_model, d_hidden, pdrop):
        super(Encoder, self).__init__()
        # create a module list of num_decoder DecoderBlock objects
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_hidden, num_hidden_layers, pdrop)
            for _ in range(num_encoders)
        ])
        
        # OPTIONAL
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, padding_mask):
        # pass the input through every decoder layer
        for block in self.encoder_blocks:
            x = block(x, padding_mask)
        
        # optional: return the output after layer normalization  
        return self.layer_norm(x)
        

# wrapper for the entire top to bottom model       
class ETPOL(nn.Module):
    """there are probably more hyper params to add here"""
    def __init__(self, vocab_size, context_length, d_model, d_hidden, num_hidden_layers, num_heads, num_encoders, pdrop):
        super(ETPOL, self).__init__()
        # create the simple embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        # init our pe object
        self.positional_encoder = PositionalEncoder(context_length, d_model)
        self.dropout = nn.Dropout(pdrop)
        
        # encoder block with MHattention + add/norm + ffn 
        self.encoder = Encoder(
            d_model=d_model,
            d_hidden=d_hidden,
            num_heads=num_heads,
            num_hidden_layers=num_hidden_layers,
            num_encoders=num_encoders,
            pdrop=pdrop
        )
        # final output projection from cls embedding to class probabilities
        self.to_logits = nn.Linear(d_model, 2)

        
    def forward(self, x):
        # ensures model disregards the padding token '1'
        input_key_mask = x == 1
        
        # everything is handled modularly, so this forward pass is smooth
        x = self.embedding_layer(x)
        x = self.dropout(x)
        x = self.positional_encoder(x)
        x = self.dropout(x)
        x = self.encoder(x, padding_mask=input_key_mask)
        
        # extract the cls token embedding, (the first in the sequence)
        cls_embedding = x[:, 0, :]
        
        # extract and return logits
        logits = self.to_logits(cls_embedding)
        
        return logits