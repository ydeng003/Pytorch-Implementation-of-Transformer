#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:09:49 2019

@author: cathy
"""

import torch
import torch.nn as nn
from sublayer import WordEmbedder,  MultiHeadAttention, FeedForward,PositionalEncoder
import numpy as np

class DecoderCycle(nn.Module):
    """
    Decoder Layers of Transformer
    
    """
    def __init__(self, tgt_vocab_size = 37000, heads = 8, d_model = 512, dropout = 0.1, max_seq_len = 80, batch_size = 64, d_ff=2048):
        super().__init__() 
        self.heads = heads
        self.d_model = d_model
        self.embedder = WordEmbedder(
                vocab_size = tgt_vocab_size, 
                d_model = self.d_model)       
        self.pos_encoder = PositionalEncoder(
                d_model = self.d_model, 
                max_seq_len = max_seq_len, 
                batch_size = batch_size, 
                dropout = dropout)
        self.maskMultiAttn = MultiHeadAttention(
                heads = self.heads, 
                d_model = self.d_model, 
                dropout = dropout)
        self.multiAttn = MultiHeadAttention(
                heads = self.heads, 
                d_model = self.d_model, 
                dropout = dropout)
        self.ff = FeedForward(
                d_model = self.d_model, 
                d_ff=d_ff, 
                dropout = dropout)
        self.linear = nn.Linear(
                in_features = d_model, 
                out_features = tgt_vocab_size)
        self.softmax = nn.Softmax(dim = -1)
        # Share the weight matrix between target word embedding & the final logit dense layer
        #self.linear.weight = self.embedder.embed.weight
        
    def forward(self, x, k, v, N, mask, iteration):
        tokens = self.embedder(input_word = x)
        tokens = self.pos_encoder(sentence = x, x = tokens)   
        # stack of 6 identical layers
        for i in range(N):
            q = self.maskMultiAttn(q = tokens, k = k, v = v, mask = mask)
            output = self.multiAttn(q = q,k = k,v = v, mask = None)
            output = self.ff(x = output)  
        output = self.linear(output)
        output = output[:,iteration]
        output = self.softmax(output)
        return output


class Decoder(nn.Module):
    """
    Decoder of Transformer    
    When training, each time send one more words from target sentence to self.decoderCycle
    When testing, each time send all the predicted words to self.decoderCycle
    
    """    
    def __init__(self, tgt_vocab_size = 37000, heads = 8, d_model = 512, dropout = 0.1, max_seq_len = 80, batch_size = 64, d_ff = 2048):
        super().__init__() 
        self.heads = heads
        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size
        self.batch_size = batch_size
        self.decoderCycle = DecoderCycle(
                tgt_vocab_size = self.tgt_vocab_size, 
                heads = self.heads, 
                d_model = self.d_model, 
                dropout = dropout, 
                max_seq_len = max_seq_len, 
                batch_size = self.batch_size, 
                d_ff = d_ff)
        self.max_seq_len = max_seq_len
        self.mask = np.triu(np.ones((self.max_seq_len,self.max_seq_len)),k = 1).astype('bool')
        self.mask = torch.from_numpy(self.mask)
    def forward(self, x, k, v, N = 6, tag = "train", eos = 2, sos = 1):
        if tag == "train":
            decoderInput = x
        else:
            decoderInput = torch.ones(self.batch_size, self.max_seq_len, dtype = torch.long).cuda()
            decoderInput[:,0] = sos
        output = decoderInput[:,0].unsqueeze(0)
        outputProb = []
        prob0 = torch.zeros(self.batch_size, self.tgt_vocab_size, dtype = torch.float).cuda()
        prob0[:, 2] = 1
        outputProb.append(prob0)
        # For iteration i, the output of self.decoderCycle is the ith predicted word
        for i in list(range(self.max_seq_len-1)):
            cycleOutput = self.decoderCycle(
                    x = decoderInput, 
                    k = k, 
                    v = v, 
                    N = N, 
                    mask = self.mask[i], 
                    iteration = i)
            outputProb.append(cycleOutput)
            _, ind = cycleOutput.max(dim = -1)
            output = torch.cat([output, ind.unsqueeze(0)])
            if tag != "train":
                decoderInput = torch.cat([output, torch.ones(self.max_seq_len - i - 2, self.batch_size, dtype = torch.long).cuda()]).T.cuda()
        output = output.T        
        outputProb = torch.stack(outputProb).transpose(0,1).transpose(1,2)
        return outputProb, output
        
    
