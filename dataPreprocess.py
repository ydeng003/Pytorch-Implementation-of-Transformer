#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:30:35 2019

@author: cathy
"""

import pandas as pd
import json
import nltk

src_data='./train.en' # original language
trg_data='./train.de' # target language
src_lang='en'
trg_lang='de'
T=80

# load data
opt_src_data = open(src_data).read().strip().split('\n')
opt_trg_data = open(trg_data).read().strip().split('\n')


# load vocabulary dictionoary for English and Germany, and save them to a Json file
dict_org = {}
i=0
with open("vocab.50K.en") as f:
    for line in f:   
        dict_org[line[:-1]] = i
        i=i+1
dict_org['<pad>'] = 50000
with open('en.json', 'w') as fp:
    json.dump(dict_org, fp)


dict_trg = {}
i=0
with open("vocab.50K.de") as f:
    for line in f:   
        dict_trg[line[:-1]] = i
        i=i+1
dict_trg['<pad>'] = 5000
with open('de.json', 'w') as fp:
    json.dump(dict_trg, fp)

# Tokenize sentence
words_org = list(map(lambda x: nltk.word_tokenize(x.lower()), opt_src_data))
words_trg = list(map(lambda x: nltk.word_tokenize(x.lower()), opt_trg_data))

# map each token with the dictionary, generate a number vector for each sentence
words_org_num = list(map(lambda x: list(map(dict_org.get, x)), words_org))
words_trg_num = list(map(lambda x: list(map(dict_trg.get, x)), words_trg))

# filter data with original sentence longer than T, and target sentence longer than T-2
org_trg_num = list(zip(words_org_num,words_trg_num))
org_trg_num =  [ elem for elem in org_trg_num if (len(elem[0]) <= T) & (len(elem[1]) <= T-2)]
words_org_num1, words_trg_num1 = map(list, zip(*org_trg_num))

def noise(x):
    if x is None:
        return 0
    else:
        return x

# Filter noise, if a token is not included in the dictionary, treat it as noise, which is 0 in the dictionary
words_org_num2 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_org_num1))
words_trg_num2 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_trg_num1))

# pad the sentence
# if the length is less than T, pad it with 50000
org_vectors = list(map(lambda x: x + [50000]*(T-len(x)), words_org_num2))
trg_vectors = list(map(lambda x: [1] + x + [2] + [50000]*(T - 2 -len(x)), words_trg_num2))

# Save org_vectors and trg_vectors to csv files
from tqdm import tqdm
with open("orig_all.csv", "w") as f:
    for row in tqdm(org_vectors):
        f.write("%s\n" % ','.join(str(col) for col in row))

from tqdm import tqdm
with open("trg_all.csv", "w") as f:
    for row in tqdm(trg_vectors):
        f.write("%s\n" % ','.join(str(col) for col in row))

# read org_vectors and trg_vectors to csv files
orig_all = pd.read_csv("orig_all.csv", header=None, names=list(range(80))) 
trg_all = pd.read_csv("trg_all.csv", header=None, names=list(range(80))) 

# split orig_all and trg_all to train, validation, test dataset.
orig_train_all = orig_all[:4000000]
orig_train_all = orig_train_all.reset_index(drop = True)
orig_train_all.to_csv('orig_train_all.csv', index = True)
orig_eval_all = orig_all[4000000:4360000]
orig_eval_all = orig_eval_all.reset_index(drop = True)
orig_eval_all.to_csv('orig_eval_all.csv', index = True)
orig_val_all = orig_all[4360000:4365000]
orig_val_all = orig_val_all.reset_index(drop = True)
orig_val_all.to_csv('orig_val_all.csv', index = True)


trg_train_all = trg_all[:4000000]
trg_train_all = trg_train_all.reset_index(drop = True)
trg_train_all.to_csv('trg_train_all.csv', index = True)
trg_eval_all = trg_all[4000000:4360000]
trg_eval_all = trg_eval_all.reset_index(drop = True)
trg_eval_all.to_csv('trg_eval_all.csv', index = True)
trg_val_all = trg_all[4360000:4365000]
trg_val_all = trg_val_all.reset_index(drop = True)
trg_val_all.to_csv('trg_val_all.csv', index = True)


"""
Generate data with length less than 10

"""
org_trg_num10 = [elem for elem in org_trg_num if (len(elem[0]) <= 10) & (len(elem[1]) <= 8)]
words_org_num10,words_trg_num10 = map(list, zip(*org_trg_num10))
words_org_num10 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_org_num10))
words_trg_num10 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_trg_num10))
org_vectors10 = list(map(lambda x: x + [50000]*(10-len(x)), words_org_num10))
trg_vectors10 = list(map(lambda x: [1] + x + [2] + [50000]*(10 - 2 -len(x)), words_trg_num10))

from tqdm import tqdm
with open("orig_all10.csv", "w") as f:
    for row in tqdm(org_vectors10):
        f.write("%s\n" % ','.join(str(col) for col in row))

from tqdm import tqdm
with open("trg_all10.csv", "w") as f:
    for row in tqdm(trg_vectors10):
        f.write("%s\n" % ','.join(str(col) for col in row))

orig_all10 = pd.read_csv("orig_all10.csv", header=None, names=list(range(10))) 
trg_all10 = pd.read_csv("trg_all10.csv", header=None, names=list(range(10))) 

orig_train10 = orig_all10[:100000]
orig_train10 = orig_train10.reset_index(drop = True)
orig_train10.to_csv('orig_train10.csv', index = True)
orig_eval10 = orig_all10[100000:110000]
orig_eval10 = orig_eval10.reset_index(drop = True)
orig_eval10.to_csv('orig_eval10.csv', index = True)
orig_val10 = orig_all10[110000:115000]
orig_val10 = orig_val10.reset_index(drop = True)
orig_val10.to_csv('orig_val10.csv', index = True)

trg_train10 = trg_all10[:100000]
trg_train10 = trg_train10.reset_index(drop = True)
trg_train10.to_csv('trg_train10.csv', index = True)
trg_eval10 = trg_all10[100000:110000]
trg_eval10 = trg_eval10.reset_index(drop = True)
trg_eval10.to_csv('trg_eval10.csv', index = True)
trg_val10 = trg_all10[110000:115000]
trg_val10 = trg_val10.reset_index(drop = True)
trg_val10.to_csv('trg_val10.csv', index = True)



"""
Generate data with length less than 20 and greater than 10

"""

org_trg_num20 = [elem for elem in org_trg_num if (len(elem[0]) <= 20) & (len(elem[1]) <= 18) & (len(elem[0]) > 10) & (len(elem[1]) > 8)]
words_org_num20,words_trg_num20 = map(list, zip(*org_trg_num20))
words_org_num20 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_org_num20))
words_trg_num20 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_trg_num20))
org_vectors20 = list(map(lambda x: x + [50000]*(20-len(x)), words_org_num20))
trg_vectors20 = list(map(lambda x: [1] + x + [2] + [50000]*(20 - 2 -len(x)), words_trg_num20))


from tqdm import tqdm
with open("orig_all20.csv", "w") as f:
    for row in tqdm(org_vectors20):
        f.write("%s\n" % ','.join(str(col) for col in row))

from tqdm import tqdm
with open("trg_all20.csv", "w") as f:
    for row in tqdm(trg_vectors20):
        f.write("%s\n" % ','.join(str(col) for col in row))


orig_all20 = pd.read_csv("orig_all20.csv", header=None, names=list(range(20))) 
trg_all20 = pd.read_csv("trg_all20.csv", header=None, names=list(range(20))) 

orig_train20 = orig_all20[:100000]
orig_train20 = orig_train20.reset_index(drop = True)
orig_train20.to_csv('orig_train20.csv', index = True)
orig_eval20 = orig_all20[100000:110000]
orig_eval20 = orig_eval20.reset_index(drop = True)
orig_eval20.to_csv('orig_eval20.csv', index = True)
orig_val20 = orig_all20[110000:115000]
orig_val20 = orig_val20.reset_index(drop = True)
orig_val20.to_csv('orig_val20.csv', index = True)

trg_train20 = trg_all20[:100000]
trg_train20 = trg_train20.reset_index(drop = True)
trg_train20.to_csv('trg_train20.csv', index = True)
trg_eval20 = trg_all20[100000:110000]
trg_eval20 = trg_eval20.reset_index(drop = True)
trg_eval20.to_csv('trg_eval20.csv', index = True)
trg_val20 = trg_all20[110000:115000]
trg_val20 = trg_val20.reset_index(drop = True)
trg_val20.to_csv('trg_val20.csv', index = True)


"""
Generate data with length less than 30 and greater than 20

"""

org_trg_num30 = [elem for elem in org_trg_num if (len(elem[0]) <= 30) & (len(elem[1]) <= 28) & (len(elem[0]) > 20) & (len(elem[1]) > 18)]
words_org_num30,words_trg_num30 = map(list, zip(*org_trg_num30))
words_org_num30 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_org_num30))
words_trg_num30 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_trg_num30))
org_vectors30 = list(map(lambda x: x + [50000]*(30-len(x)), words_org_num30))
trg_vectors30 = list(map(lambda x: [1] + x + [2] + [50000]*(30 - 2 -len(x)), words_trg_num30))


from tqdm import tqdm
with open("orig_all30.csv", "w") as f:
    for row in tqdm(org_vectors30):
        f.write("%s\n" % ','.join(str(col) for col in row))

from tqdm import tqdm
with open("trg_all30.csv", "w") as f:
    for row in tqdm(trg_vectors30):
        f.write("%s\n" % ','.join(str(col) for col in row))


orig_all30 = pd.read_csv("orig_all30.csv", header=None, names=list(range(30))) 
trg_all30 = pd.read_csv("trg_all30.csv", header=None, names=list(range(30))) 

orig_train30 = orig_all30[:100000]
orig_train30 = orig_train30.reset_index(drop = True)
orig_train30.to_csv('orig_train30.csv', index = True)
orig_eval30 = orig_all30[100000:110000]
orig_eval30 = orig_eval30.reset_index(drop = True)
orig_eval30.to_csv('orig_eval30.csv', index = True)
orig_val30 = orig_all30[110000:115000]
orig_val30 = orig_val30.reset_index(drop = True)
orig_val30.to_csv('orig_val30.csv', index = True)

trg_train30 = trg_all30[:100000]
trg_train30 = trg_train30.reset_index(drop = True)
trg_train30.to_csv('trg_train30.csv', index = True)
trg_eval30 = trg_all30[100000:110000]
trg_eval30 = trg_eval30.reset_index(drop = True)
trg_eval30.to_csv('trg_eval30.csv', index = True)
trg_val30 = trg_all30[110000:115000]
trg_val30 = trg_val30.reset_index(drop = True)
trg_val30.to_csv('trg_val30.csv', index = True)

"""
Generate data with length less than 40 and greater than 30

"""

org_trg_num40 = [elem for elem in org_trg_num if (len(elem[0]) <= 40) & (len(elem[1]) <= 38) & (len(elem[0]) > 30) & (len(elem[1]) > 28)]
words_org_num40,words_trg_num40 = map(list, zip(*org_trg_num40))
words_org_num40 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_org_num40))
words_trg_num40 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_trg_num40))
org_vectors40 = list(map(lambda x: x + [50000]*(40-len(x)), words_org_num40))
trg_vectors40 = list(map(lambda x: [1] + x + [2] + [50000]*(40 - 2 -len(x)), words_trg_num40))

from tqdm import tqdm
with open("orig_all40.csv", "w") as f:
    for row in tqdm(org_vectors40):
        f.write("%s\n" % ','.join(str(col) for col in row))

from tqdm import tqdm
with open("trg_all40.csv", "w") as f:
    for row in tqdm(trg_vectors40):
        f.write("%s\n" % ','.join(str(col) for col in row))

orig_all40 = pd.read_csv("orig_all40.csv", header=None, names=list(range(40))) 
trg_all40 = pd.read_csv("trg_all40.csv", header=None, names=list(range(40))) 

orig_train40 = orig_all40[:100000]
orig_train40 = orig_train40.reset_index(drop = True)
orig_train40.to_csv('orig_train40.csv', index = True)
orig_eval40 = orig_all40[100000:110000]
orig_eval40 = orig_eval40.reset_index(drop = True)
orig_eval40.to_csv('orig_eval40.csv', index = True)
orig_val40 = orig_all40[110000:115000]
orig_val40 = orig_val40.reset_index(drop = True)
orig_val40.to_csv('orig_val40.csv', index = True)

trg_train40 = trg_all40[:100000]
trg_train40 = trg_train40.reset_index(drop = True)
trg_train40.to_csv('trg_train40.csv', index = True)
trg_eval40 = trg_all40[100000:110000]
trg_eval40 = trg_eval40.reset_index(drop = True)
trg_eval40.to_csv('trg_eval40.csv', index = True)
trg_val40 = trg_all40[110000:115000]
trg_val40 = trg_val40.reset_index(drop = True)
trg_val40.to_csv('trg_val40.csv', index = True)

"""
Generate data with length less than 60 and greater than 40

"""

org_trg_num60 = [elem for elem in org_trg_num if (len(elem[0]) <= 60) & (len(elem[1]) <= 58) & (len(elem[0]) > 40) & (len(elem[1]) > 38)]
words_org_num60,words_trg_num60 = map(list, zip(*org_trg_num60))
words_org_num60 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_org_num60))
words_trg_num60 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_trg_num60))
org_vectors60 = list(map(lambda x: x + [50000]*(60-len(x)), words_org_num60))
trg_vectors60 = list(map(lambda x: [1] + x + [2] + [50000]*(60 - 2 -len(x)), words_trg_num60))

from tqdm import tqdm
with open("orig_all60.csv", "w") as f:
    for row in tqdm(org_vectors60):
        f.write("%s\n" % ','.join(str(col) for col in row))

from tqdm import tqdm
with open("trg_all60.csv", "w") as f:
    for row in tqdm(trg_vectors60):
        f.write("%s\n" % ','.join(str(col) for col in row))

orig_all60 = pd.read_csv("orig_all60.csv", header=None, names=list(range(60))) 
trg_all60 = pd.read_csv("trg_all60.csv", header=None, names=list(range(60))) 

orig_train60 = orig_all60[:100000]
orig_train60 = orig_train60.reset_index(drop = True)
orig_train60.to_csv('orig_train60.csv', index = True)
orig_eval60 = orig_all60[100000:110000]
orig_eval60 = orig_eval60.reset_index(drop = True)
orig_eval60.to_csv('orig_eval60.csv', index = True)
orig_val60 = orig_all60[110000:115000]
orig_val60 = orig_val60.reset_index(drop = True)
orig_val60.to_csv('orig_val60.csv', index = True)

trg_train60 = trg_all60[:100000]
trg_train60 = trg_train60.reset_index(drop = True)
trg_train60.to_csv('trg_train60.csv', index = True)
trg_eval60 = trg_all60[100000:110000]
trg_eval60 = trg_eval60.reset_index(drop = True)
trg_eval60.to_csv('trg_eval60.csv', index = True)
trg_val60 = trg_all60[110000:115000]
trg_val60 = trg_val60.reset_index(drop = True)
trg_val60.to_csv('trg_val60.csv', index = True)

"""
Generate data with length less than 80 and greater than 60

"""

org_trg_num80 = [elem for elem in org_trg_num if (len(elem[0]) <= 80) & (len(elem[1]) <= 78) & (len(elem[0]) > 60) & (len(elem[1]) > 58)]
words_org_num80,words_trg_num80 = map(list, zip(*org_trg_num80))
words_org_num80 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_org_num80))
words_trg_num80 = list(map(lambda x: list(map(lambda item: noise(item), x)), words_trg_num80))
org_vectors80 = list(map(lambda x: x + [50000]*(80-len(x)), words_org_num80))
trg_vectors80 = list(map(lambda x: [1] + x + [2] + [50000]*(80 - 2 -len(x)), words_trg_num80))

from tqdm import tqdm
with open("orig_all80.csv", "w") as f:
    for row in tqdm(org_vectors80):
        f.write("%s\n" % ','.join(str(col) for col in row))

from tqdm import tqdm
with open("trg_all80.csv", "w") as f:
    for row in tqdm(trg_vectors80):
        f.write("%s\n" % ','.join(str(col) for col in row))

orig_all80 = pd.read_csv("orig_all80.csv", header=None, names=list(range(80))) 
trg_all80 = pd.read_csv("trg_all80.csv", header=None, names=list(range(80))) 

orig_train80 = orig_all80[:50000]
orig_train80 = orig_train80.reset_index(drop = True)
orig_train80.to_csv('orig_train80.csv', index = True)
orig_eval80 = orig_all80[50000:55000]
orig_eval80 = orig_eval80.reset_index(drop = True)
orig_eval80.to_csv('orig_eval80.csv', index = True)
orig_val80 = orig_all80[55000:55500]
orig_val80 = orig_val80.reset_index(drop = True)
orig_val80.to_csv('orig_val80.csv', index = True)

trg_train80 = trg_all80[:50000]
trg_train80 = trg_train80.reset_index(drop = True)
trg_train80.to_csv('trg_train80.csv', index = True)
trg_eval80 = trg_all80[50000:55000]
trg_eval80 = trg_eval80.reset_index(drop = True)
trg_eval80.to_csv('trg_eval80.csv', index = True)
trg_val80 = trg_all80[55000:55500]
trg_val80 = trg_val80.reset_index(drop = True)
trg_val80.to_csv('trg_val80.csv', index = True)

