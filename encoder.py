#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:09:49 2019

@author: cathy
"""

import torch.nn as nn
from sublayer import WordEmbedder, MultiHeadAttention, FeedForward, PositionalEncoder


class Encoder(nn.Module):
    """
    Encoder Layers of Transformer
    
    """
    def __init__(self, src_vocab_size = 37000, heads = 8, d_model = 512, dropout = 0.1, max_seq_len = 80, batch_size = 64, d_ff=2048):
        super().__init__() 
        self.heads = heads
        self.d_model = d_model
        self.embedder = WordEmbedder(
                vocab_size = src_vocab_size, 
                d_model = self.d_model)
        self.pos_encoder = PositionalEncoder(
                d_model = self.d_model, 
                max_seq_len = max_seq_len, 
                batch_size = batch_size, 
                dropout = dropout)
        self.multiAttn = MultiHeadAttention(
                heads = self.heads, 
                d_model = self.d_model, 
                dropout = dropout)
        self.ff = FeedForward(
                d_model = self.d_model, 
                d_ff=d_ff, 
                dropout = dropout)
    
    def forward(self, x, N = 6, mask = None):
        tokens = self.embedder(input_word = x)
        tokens = self.pos_encoder(sentence = x, x = tokens)    
        # stack of 6 identical layers
        for i in range(N):
            output = self.multiAttn(q = tokens, k = tokens, v = tokens, mask = mask)
            output = self.ff(x = output)            
            tokens = output            
        return output 
    
