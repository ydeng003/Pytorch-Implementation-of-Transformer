#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:28:55 2019

@author: cathy
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import time
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from optimScheduler import OptimScheduler
from torch import optim
from transformer import Transformer
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class TransformerDataset(Dataset):

    def __init__(self, orig_csv, tgt_csv):
        """
        Prepare Dataset
        Args:
            orig_csv (string): Path to the original language csv.
            tgt_csv (string): Path to the target language csv.
        """
        self.features = pd.read_csv(orig_csv, index_col = 0)
        self.targets = pd.read_csv(tgt_csv, index_col = 0)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        features = self.features.iloc[idx, :].T.values.tolist()
        targets = self.targets.iloc[idx, :].T.values.tolist()
        return features, targets
    



def dataloaders(train_data, valid_data, batch_size):
    """
    Prepare DataLoader
    
    """
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)

    valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle=True)
    return train_loader, valid_loader

    
    
def train_epoch(model, training_data, optimizer, N, device, epoch, opt):
    
    # initialize the training log file
    if opt.log:
        log_train_file = opt.log + '.train.log'
    
    model.train()
    total_loss = 0
    total_correct = 0
    n_word = 0
    count = 0
    start = time.time()
    batch_counter = 1
    loss_ls = []
    accu_ls = []
    batch_accuracy = 0
    batch_loss = 0
    for batch in training_data: # Get Batch
        
        orig, tgt = batch
        orig = torch.stack(orig).T.type(torch.long)
        tgt = torch.stack(tgt).T.type(torch.long)
        orig = orig.cuda()
        tgt = tgt.cuda()
        
        outputProb, pred = model(orig, tgt, N, tag = "train")  # Pass Batch
        loss = F.cross_entropy(outputProb, tgt)  # Calculate Loss
        
        # zero out weight when the model has trained 10 batches
        if (batch_counter % 10 == 0) and (batch_counter != 0):
            optimizer.zero_grad()
            
        # update weights when the model has trained 10 batches
        loss.backward() # Calculate Gradients
        if (batch_counter % 10 == 0) and (batch_counter != 0):
            optimizer.step_and_update_lr() # Update Weights
        
        # total_correct and n_word is used to calculate accuracy of an epoch
        total_correct += torch.sum((pred == tgt).int()).item()
        n_word += np.prod(tgt.shape)
        
        # total_loss is the loss of an epoch
        total_loss += loss.item()
        
        # batch_accuracy is accuray of 10 batches
        batch_accuracy += torch.sum((pred == tgt).int()).item() / np.prod(tgt.shape)
        batch_loss += loss.item()
        
        # log epoch, batch number, batch loss, batch accuracy, elapse in train log file
        if (batch_counter % 10 == 0) and (batch_counter != 0):
            if log_train_file:
                with open(log_train_file, 'a') as log_tf:
                    log_tf.write('epoch: {epoch},batch: {batch_counter}, loss: {loss},accuracy: {accu},elapses: {elapse:3.3f}\n'.format(
                            epoch=epoch, batch_counter = batch_counter // 10, loss=batch_loss / 10,
                            accu=batch_accuracy / 10, elapse=(time.time()-start)/60))
            loss_ls.append(batch_loss / 10)
            accu_ls.append(batch_accuracy / 10)
            batch_accuracy = 0
            batch_loss = 0
            start = time.time()
        
        
        if count == 0:
            preds = pred
            count += 1
        else:
            preds = torch.cat([preds, pred])
        batch_counter += 1
        

    accuracy = total_correct/n_word
    loss_per_case = total_loss/len(training_data)
    
    return preds, loss_per_case, accuracy, loss_ls, accu_ls

def eval_epoch(model, validation_data, N, device, epoch, opt):
    
    # initialize the validation log file
    if opt.log:
        log_valid_file = opt.log + '.valid.log'
        
    model.eval()
    total_loss = 0
    total_correct = 0
    n_word = 0
    count = 0
    start = time.time()
    batch_counter = 1
    loss_ls = []
    accu_ls = []
    
    for batch in validation_data: # Get Batch
        
        orig, tgt = batch
        orig = torch.stack(orig).T.type(torch.long)
        tgt = torch.stack(tgt).T.type(torch.long)
        orig = orig.cuda()
        tgt = tgt.cuda()
        
        outputProb, pred = model(orig, tgt, N, tag = "valid") # Pass Batch
        loss = F.cross_entropy(outputProb, tgt)  # Calculate Loss
        
        # total_correct and n_word is used to calculate accuracy of an epoch
        total_correct += torch.sum((pred == tgt).int()).item()
        n_word += np.prod(tgt.shape)
        
        # total_loss is the loss of an epoch
        total_loss += loss.item()
        
        # batch_accuracy is accuray of each batche
        batch_accuracy = torch.sum((pred == tgt).int()).item() / np.prod(tgt.shape)
        
        # log epoch, batch number, batch loss, batch accuracy, elapse in train log file
        if log_valid_file:
            with open(log_valid_file, 'a') as log_tf:
                log_tf.write('epoch: {epoch},batch: {batch_counter}, loss: {loss},accuracy: {accu},elapses: {elapse:3.3f}\n'.format(
                    epoch=epoch, batch_counter = batch_counter, loss=loss.item(),
                    accu=batch_accuracy, elapse=(time.time()-start)/60))
                
        loss_ls.append(loss.item())
        accu_ls.append(batch_accuracy)
        
        if count == 0:
            preds = pred
            count += 1
        else:
            preds = torch.cat([preds, pred])
        batch_counter += 1
        
    accuracy = total_correct/n_word
    loss_per_case = total_loss/len(validation_data)
    return preds, loss_per_case, accuracy, loss_ls, accu_ls


def train(model, training_data, validation_data, optimizer, device, opt):
    
    # initialize the train / validation log file
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch, loss, accuracy, elapse\n')
            log_vf.write('epoch, loss, accuracy, elapse\n')
    
    train_loss_epoch_ls = []
    train_accu_epoch_ls = []
    valid_loss_epoch_ls = []
    valid_accu_epoch_ls = []
    train_loss_batch_ls = []
    train_accu_batch_ls = []
    valid_loss_batch_ls = []
    valid_accu_batch_ls = []
    print("Number of epochs: ", opt.epochs)
    
    for epoch in range(opt.epochs):
        
        start = time.time()
        
        train_preds, train_loss, train_accu, train_loss_batch, train_accu_batch = train_epoch(model, training_data, optimizer, opt.n_layers, device, epoch, opt)
        
        print('epoch:', epoch, 'training loss:', train_loss, 'training accuracy:',train_accu, 
              'elapse: {elapse:3.3f} min'.format(elapse=(time.time()-start)/60))
        if log_train_file:
            with open(log_train_file, 'a') as log_tf:
                log_tf.write('epoch: {epoch},loss: {loss},accuracy: {accu},elapses: {elapse:3.3f}\n'.format(
                    epoch=epoch, loss=train_loss,
                    accu=train_accu, elapse=(time.time()-start)/60))
        
        train_loss_epoch_ls.append(train_loss)
        train_accu_epoch_ls.append(train_accu)
        train_loss_batch_ls.append(train_loss_batch)
        train_accu_batch_ls.append(train_accu_batch)
        
        start = time.time()
        
        valid_preds, valid_loss, valid_accu, valid_loss_batch, valid_accu_batch = eval_epoch(model, validation_data, opt.n_layers, device, epoch, opt)
        
        print('epoch:', epoch, 'validation loss:', valid_loss, 'validation accuracy:',valid_accu, 
              'elapse: {elapse:3.3f} min'.format(elapse=(time.time()-start)/60))
        if log_valid_file:
            with open(log_valid_file, 'a') as log_vf:
                log_vf.write('epoch: {epoch},loss: {loss},accuracy: {accu},elapses: {elapse:3.3f}\n'.format(
                    epoch=epoch, loss=valid_loss,
                    accu=valid_accu, elapse=(time.time()-start)/60))
        
        valid_loss_epoch_ls.append(valid_loss)
        valid_accu_epoch_ls.append(valid_accu)
        valid_loss_batch_ls.append(valid_loss_batch)
        valid_accu_batch_ls.append(valid_accu_batch)       
        
        # save model after training an epoch
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch}
        if opt.save_model:
            model_name = 'epoch_' + str(epoch) + '_model.chkpt'
            torch.save(checkpoint, model_name)
            print('    - [Info] epoch_',epoch,' checkpoint file has been saved.')
            # save the best model
            best_model_name = opt.save_model + '.chkpt'
            if valid_accu >= max(valid_accu_epoch_ls):
                torch.save(checkpoint, best_model_name)
                print('    - [Info] The checkpoint file has been updated.')
    
    return train_preds, train_loss_epoch_ls, train_accu_epoch_ls, train_loss_batch_ls, train_accu_batch_ls, valid_preds, valid_loss_epoch_ls, valid_accu_epoch_ls, valid_loss_batch_ls, valid_accu_batch_ls

            

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-train_orig', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_orig', required=True)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-orig_dict', required=True)
    parser.add_argument('-tgt_dict', required=True)
    
    parser.add_argument('-train_steps', type=int, default=100000)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-max_seq_len', type=int, default=80) 
    
    parser.add_argument('-src_vocab_size', type=int, default=13725)
    parser.add_argument('-tgt_vocab_size', type=int, default=23472)   
    
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_ff', type=int, default=2048)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-no_cuda', action='store_true')
    
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    
    # read training data and validation data
    train_data = TransformerDataset(orig_csv = opt.train_orig, tgt_csv = opt.train_tgt)
    valid_data = TransformerDataset(orig_csv = opt.valid_orig, tgt_csv = opt.valid_tgt)
    train_tgts = pd.read_csv(opt.train_tgt, index_col = 0)
    train_tgts = train_tgts.values
    train_tgts = torch.from_numpy(train_tgts)
    valid_tgts = pd.read_csv(opt.valid_tgt, index_col = 0)
    valid_tgts = valid_tgts.values
    valid_tgts = torch.from_numpy(valid_tgts)
    training_data, validation_data = dataloaders(train_data, valid_data, opt.batch_size)
    
    # load dictionary file, calculate the vocabulary size
    with open(opt.orig_dict) as json_file:
        orig_dict = json.load(json_file)
    opt.src_vocab_size = len(orig_dict)
    with open(opt.tgt_dict) as json_file:
        tgt_dict = json.load(json_file)
    opt.tgt_vocab_size = len(tgt_dict)
    
    print(opt)
    #print(opt.tgt_vocab_size)
    
    # Initialize model
    model = Transformer(
            src_vocab_size = opt.src_vocab_size, 
            tgt_vocab_size = opt.tgt_vocab_size, 
            heads = opt.n_head, 
            d_model = opt.d_model, 
            dropout = opt.dropout, 
            max_seq_len = opt.max_seq_len, 
            batch_size = opt.batch_size,
            d_ff = opt.d_ff)

    # send model to GPU
    model.to('cuda')
    
    # Initialize optimizer
    optimizer = OptimScheduler(optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
            d_model=opt.d_model, warmup_steps=opt.n_warmup_steps)
    
    # Initialize device
    device = torch.device('cuda')
    
    # Calculate number of epoch
    opt.epochs = opt.train_steps//(len(train_data) //(opt.batch_size * 10))

    # train the model
    train_preds, train_loss_epoch_ls, train_accu_epoch_ls, train_loss_batch_ls, train_accu_batch_ls, valid_preds, valid_loss_epoch_ls, valid_accu_epoch_ls, valid_loss_batch_ls, valid_accu_batch_ls = train(model, training_data, validation_data, optimizer, device, opt)
                        
    # plot the loss line chart  
    n_epochs = list(range(1,len(train_loss_epoch_ls)+1))
    loss_plot = plt.figure(1)
    plt.plot(n_epochs, train_loss_epoch_ls, marker='o', markersize=5, color='r', linewidth=1, label = 'train')
    plt.plot(n_epochs, valid_loss_epoch_ls, marker='o', color='olive', linewidth=1, label = 'validation')
    plt.title('Loss line chart')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    loss_plot.savefig('loss.png')
    
    # plot the accuracy line chart
    accu_plot = plt.figure(2)
    plt.plot(n_epochs, train_accu_epoch_ls, marker='o', markersize=5, color='r', linewidth=1, label = 'train')
    plt.plot(n_epochs, valid_accu_epoch_ls, marker='o', color='olive', linewidth=1, label = 'validation')
    plt.title('accuracy line chart')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    accu_plot.savefig('accuracy.png')
    
    # save batch loss and batch accuracy to txt files
    with open("train_loss_batch.txt", "w") as tl, open("train_accu_batch.txt", "w") as ta, open("valid_loss_batch.txt", "w") as vl, open("valid_accu_batch.txt", "w") as va:
        tl.write(str(train_loss_batch_ls))
        ta.write(str(train_accu_batch_ls))
        vl.write(str(valid_loss_batch_ls))
        va.write(str(valid_accu_batch_ls))
    
    # save train set prediction and validation set prediction    
    torch.save(train_preds, 'train_preds.pt')
    torch.save(train_tgts, 'train_tgts.pt')
    torch.save(valid_preds, 'valid_preds.pt')
    torch.save(valid_tgts, 'valid_tgts.pt')
    
if __name__ == '__main__':
    main()

