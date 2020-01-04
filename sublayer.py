#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 09 17:12:36 2019

@author: cathy
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math




class WordEmbedder(nn.Module):
    """
    Embed the input tokens and output tokens to vectors of dimention d_model
    
    """
    # vocab_size is the number of words in train, validation and test set
    def __init__(self, vocab_size = 37000, d_model = 512):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.vocab_size = vocab_size
        self.d_model = d_model
    def forward(self, input_word):
        output = self.embed(input_word) * math.sqrt(self.d_model) 
        # increase the embedding vector to make sure the original meaning in the embedding vector 
        # won’t be lost when it is added to positional encodings.        
        return output


class PositionalEncoder(nn.Module):
    """
    Inject the relative or absolute position information in the sequence
    The output is the addition of "positional encodings" and output of the Embedding layer
    
    """
    def __init__(self, d_model = 512, batch_size = 64, max_seq_len = 80, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = torch.ones(self.batch_size, self.max_seq_len, self.d_model).cuda()      
        for pos in range(max_seq_len):
            for i in range(0, self.d_model, 2):
                self.pos_encoder[:,pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                self.pos_encoder[:,pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
    def forward(self, sentence, x):       
        output = x.add_(self.pos_encoder)#.cuda()
        output = self.dropout(output)
        return output

class Attention(nn.Module):
    """
    Attention module, calculate attention using q, k, v generated from the input vector 
    
    """
    def __init__(self, d_k = 64, dropout = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = 2)
    def forward(self, q, k, v, mask = None):
        output = torch.matmul(q, k.transpose(2,3)) / math.sqrt(self.d_k)       
        if mask is not None:
            output = output.masked_fill(mask.cuda(), -1e9)
        output = self.softmax(output)
        output = torch.matmul(output, v)
        output = self.dropout(output)        
        return output

class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention sublayer
    
    """
    def __init__(self, heads = 8, d_model = 512, dropout = 0.1):
        super().__init__()        
        self.d_model = d_model
        self.d_q = d_model // heads
        self.d_k = d_model // heads
        self.d_v = d_model // heads
        self.heads = heads        
        self.q_mat = nn.Linear(in_features = d_model, out_features = d_model)
        self.v_mat = nn.Linear(in_features = d_model, out_features = d_model)
        self.k_mat = nn.Linear(in_features = d_model, out_features = d_model)
        self.dropout = dropout 
        self.attn = Attention(d_k = self.d_k, dropout = dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, q, k, v, mask=None):
        
        sz = q.size(1) 
        residual = q
        
        # calculate q,k,v and split into N heads
        q = self.q_mat(q).view(-1, sz, self.heads, self.d_k)
        k = self.k_mat(k).view(-1, sz, self.heads, self.d_k)        
        v = self.v_mat(v).view(-1, sz, self.heads, self.d_k)

        q = q.transpose(1,2)
        k = k.transpose(1,2)     
        v = v.transpose(1,2)

        output = self.attn(q = q, k = k, v = v, mask = mask)
        
        # concatenate outputs from multi-head attention
        output = output.transpose(1,2).contiguous().view(-1, sz, self.d_model)
        output = self.layer_norm(output + residual)
        return output   
    
    
class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Networks.
    The dimensionality of input and output is d_model
    The inner-layer's dimension is d_ff = 2048 as indicated in the paper
    
    """
    def __init__(self, d_model = 512, d_ff=2048, dropout = 0.1):
        super().__init__() 
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        output = self.l2(F.relu(self.l1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output+x)
        return output  
