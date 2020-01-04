import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from transformer import Transformer
from train_gpu_multibatch import TransformerDataset
import json
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

def BLEU(tgts, preds):
    
    preds = preds.unsqueeze(dim = 1)
    
    tgts_ls = tgts.tolist()
    preds_ls = preds.tolist()
    
    tgts_ls = list(map(lambda x: [token for token in x if (token != 50000) & (token != 1) & (token != 2)], tgts_ls))
    preds_ls = list(map(lambda x: [[token for token in x[0] if (token != 50000) & (token != 1) & (token != 2)]], preds_ls))

    preds_tgts_ls = list(zip(preds_ls,tgts_ls))
    BleuScore_ls = np.array(list(map(lambda x: sentence_bleu(x[0],x[1], weights=(0.5, 0.5, 0, 0)), preds_tgts_ls)))
    BleuScore = np.average(BleuScore_ls) % 100
    
    return BleuScore
    
    #print("BLEU for test data:", BleuScore)
    
def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-orig_dict', required=True)
    parser.add_argument('-tgt_dict', required=True)
    parser.add_argument('-src_vocab_size', type=int, default=13725)
    parser.add_argument('-tgt_vocab_size', type=int, default=23472)
    parser.add_argument('-test_orig', required=True)
    parser.add_argument('-test_tgt', required=True)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_ff', type=int, default=2048)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-max_seq_len', type=int, default=80)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-log', default=None)
    
    opt = parser.parse_args()
    
    # load dictionary file, calculate the vocabulary size
    with open(opt.orig_dict) as json_file:
        orig_dict = json.load(json_file)
    opt.src_vocab_size = len(orig_dict)
    with open(opt.tgt_dict) as json_file:
        tgt_dict = json.load(json_file)
    opt.tgt_vocab_size = len(tgt_dict)
    
    # load testing data
    test_data = TransformerDataset(orig_csv = opt.test_orig, tgt_csv = opt.test_tgt)
    test_loader = DataLoader(test_data, batch_size = opt.batch_size, shuffle=True)
    test_tgts = pd.read_csv(opt.test_tgt, index_col = 0)
    test_tgts = test_tgts.values
    test_tgts = torch.from_numpy(test_tgts)
    
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
    checkpoint = torch.load('bestmodel.chkpt')
    model.load_state_dict(checkpoint['model'])
    opt = checkpoint['settings']
    epoch = checkpoint['epoch']
    print("epoch:", epoch)
    
    model.to('cuda')
    model.eval()

    total_loss = 0
    total_correct = 0
    n_word = 0
    count = 0
    
    # test the best model
    for batch in test_loader: # Get Batch
        orig, tgt = batch
        orig = torch.stack(orig).T.type(torch.long)
        tgt = torch.stack(tgt).T.type(torch.long)
        orig = orig.cuda()
        tgt = tgt.cuda()
        outputProb, pred = model(orig, tgt)  # Pass Batch
        loss = F.cross_entropy(outputProb, tgt)  # Calculate Loss
        total_correct += torch.sum((pred == tgt).int()).item()
        n_word += np.prod(tgt.shape)
        total_loss += loss.item()
        if count == 0:
            preds = pred
            count += 1
        else:
            preds = torch.cat([preds, pred])
            
    
    accuracy = total_correct/n_word
    #print(total_loss)
    loss_per_case = total_loss/len(test_loader) 
    print("loss per sentence: ", loss_per_case)
    print("accuracy: ", accuracy )
    
    bleuScore = BLEU(test_tgts, preds)
    print("BLEU for test data:" ,bleuScore)
    
    
    torch.save(preds, 'test_preds.pt')
    torch.save(test_tgts, 'test_tgts.pt')   
    
    
    
if __name__ == '__main__':
    main()



