#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:10:37 2019

@author: cathy
"""

#import torch
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size = 37000, tgt_vocab_size = 37000, heads = 8, d_model = 512, dropout = 0.1, max_seq_len = 80, batch_size = 64, d_ff=2048):
        super().__init__() 
        self.heads = heads
        self.d_model = d_model
        self.encoder = Encoder(
                src_vocab_size = src_vocab_size, 
                heads = self.heads, 
                d_model = self.d_model, 
                dropout = dropout, 
                max_seq_len = max_seq_len, 
                batch_size = batch_size, 
                d_ff=d_ff)
        self.decoder = Decoder(
                tgt_vocab_size = tgt_vocab_size, 
                heads = self.heads, 
                d_model = self.d_model, 
                dropout = dropout, 
                max_seq_len = max_seq_len, 
                batch_size = batch_size, 
                d_ff=d_ff)
        # Share the weight matrix between source & target word embeddings
        #self.encoder.embedder.embed.weight = self.decoder.decoderCycle.embedder.embed.weight
    
    def forward(self, x_encoder, x_decoder, N = 6, tag = "train"):
        # The key and value of decorder is from encoder. the query is from the multihead attention layer
        output_encoder = self.encoder(x = x_encoder, N = N)
        outputProb, output = self.decoder(
                x = x_decoder, 
                k = output_encoder, 
                v = output_encoder, 
                N = N, 
                tag = tag)
        return outputProb, output 
    
