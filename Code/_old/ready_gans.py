# -*- coding: utf-8 -*-
"""READY - GANs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TpGdiOui4TXQNGsdLuQSWXH_KKcR76WH

** MODEL SIZE **

<hr>

*Needs to set*
"""

# generator hidden size
model_sizes = [128, 256, 512]

"""# **GANs for Abstractive Text Summarization**
## **NLP Group Project**
## **Statistical Natural Language Processing (COMP0087), University College London**

<hr>

**Project description**

A lot of endeavours have already been devoted to NLP text summarization techniques, and abstractive methods have proved to be more proficient in generating human-like sentences. At the same time, GANs has been enjoying considerable success in the area of real-valued data such as an image generation. Recently, researchers have begun to come up with ideas on how to overcome various obstacles during training GAN models for discrete data, though not a lot of work seemed to be directly dedicated to the text summarization itself. We, therefore, would like to pursue to tackle the issue of text summarization using the GAN techniques inspired by sources enlisted below.

<hr>

**Collaborators**

- Daniel Stancl (daniel.stancl.19@ucl.ac.uk)
- Dorota Jagnesakova (dorota.jagnesakova.19@ucl.ac.uk)
- Guolinag HE (guoliang.he.19@ucl.ac.uk)
- Zakhar Borok

# **1 Setup**

<hr>

- install and import libraries
- download stopwords
- remove and clone the most recent version of git repository
- run a script with a CONTRACTION_MAP
- run a script with a function for text preprocessing

### **GitHub stuff**

**Set GitHub credentials and username of repo owner**
"""

# credentials
user_email = 'dannyi@seznam.cz'
user = "gansforlife"
user_password = "dankodorkamichaelzak"

# username of repo owner
owner_username = 'stancld'
# reponame
reponame = 'GeneratingHeadline_GANs'

# generate 
add_origin_link = (
    'https://{}:{}github@github.com/{}/{}.git'.format(
    user, user_password, owner_username, reponame)
)

print("Link used for git cooperation:\n{}".format(add_origin_link))

"""**Clone GitHub repo on the personal drive**"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# ## Clone GitHub repo to the desired folder
# # Mount drive
# from google.colab import drive
# drive.mount("/content/drive", force_remount = True)
# %cd "drive/My Drive/projects"
# 
# # Remove NLP_Project if presented and clone up-to-date repo
# !rm -r GeneratingHeadline_GANs
# !git clone https://github.com/stancld/GeneratingHeadline_GANs.git
# 
# # Go to the NLP_Project folder
# %cd GeneratingHeadline_GANs
# 
# # Config global user and add origin enabling us to execute push commands
# !git config --global user.email user_email
# !git remote rm origin
# !git remote add origin https://gansforlife:dankodorkamichaelzakgithub@github.com/stancld/GeneratingHeadline_GANs.git

"""**Function push_to_repo**"""

def push_to_repo():
  """
  models_branch
  """
  !git remote rm origin
  !git remote add origin https://gansforlife:dankodorkamichaelzak@github.com/stancld/GeneratingHeadline_GANs.git
  !git checkout master
  !git pull origin master
  !git checkout models_branch
  !git add .
  !git commit -m "model state update"
  !git checkout master
  !git merge models_branch
  !git push -u origin master

"""### **General stuff**

**Import essential libraries and load necessary conditionalities**
"""

pip install rouge

# Commented out IPython magic to ensure Python compatibility.
import os
import sys
import time
import gc
import copy
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import re
import unicodedata
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

from rouge import Rouge

# %matplotlib inline

nltk.download('stopwords')

"""**Set essential parameters**"""

# Set torch.device to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())

"""**Run python files from with classes used throughtout the document**"""

run Code/contractions.py

# code for text_preprocessing()
run Code/text_preprocessing.py

# code for transforming data to padded array
run Code/data2PaddedArray.py

# code for the generator
run Code/Models/Attention_seq2seq.py

# code for the training class (generator)
run Code/Models/generator_training_class.py

# code for the discriminator
run Code/Models/CNN_text_clf.py

# code for the training class (generator)
run Code/Models/discriminator_training_class.py

"""### **Pretrained embeddings**

<hr>

**TODO:** *Put a comment which kind of embeddings we used. Add some references and so on*
"""

embed_dim = 200

"""

# Download and unzip GloVe embedding
#!wget http://nlp.stanford.edu/data/glove.6B.zip
#!unzip glove.6B.zip


# input your pre-train txt path and parse the data
path = '../data/glove.6B.{:.0f}d.txt'.format(embed_dim)

# for michael
# path = r'/content/drive/My Drive/glove.6B.{:.0f}d.txt'.format(embed_dim)

embed_dict = {}
with open(path,'r') as f:
  lines = f.readlines()
  for l in lines:
    w = l.split()[0]
    v = np.array(l.split()[1:]).astype('float')
    embed_dict[w] = v

embed_dict['@@_unknown_@@'] = np.random.random(embed_dim)

# remove all the unnecesary files
#!rm -rf glove.6B.zip
#!rm -rf glove.6B.50d.txt
#!rm -rf glove.6B.100d.txt
#!rm -rf glove.6B.200d.txt
#!rm -rf glove.6B.300d.txt

# check the length of the dictionary
len(embed_dict.keys())

"""

"""**Function for extracting relevant matrix of pretrained weights**"""

def extract_weight(text_dictionary):
  """
  :param text_dictionary:
  """
  pre_train_weight = []
  for word_index in text_dictionary.index2word.keys():
    if word_index != 0:
      word = text_dictionary.index2word[word_index]
      try:
        word_vector = embed_dict[word].reshape(1,-1)
      except:
        word_vector = embed_dict['@@_unknown_@@'].reshape(1,-1) # handle unknown word
      pre_train_weight = np.vstack([pre_train_weight,word_vector])
    
    # add for padding
    elif word_index == len(text_dictionary.index2word.keys()):  
      pre_train_weight = np.r_[pre_train_weight, np.zeros((1, embed_dim))]
    
    else:
      word = text_dictionary.index2word[word_index]
      try:
        word_vector = embed_dict[word].reshape(1,-1)
      except:
        word_vector = embed_dict['@@_unknown_@@'].reshape(1,-1) # handle unknown word
      pre_train_weight = word_vector
  return pre_train_weight

"""# **2 Load and process the data**

<hr>

**Source of the data:** https://ucsb.app.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag

##### *Open and preprocess the data*
"""

# Commented out IPython magic to ensure Python compatibility.
# """
# 
# %%time
# # Open
# data = pd.read_csv('../data/wikihowSep.csv',
#                     error_bad_lines = False).astype(str)
# print(data.shape)
# 
# # Preprocess
# for item in ['text', 'headline']:
#   exec("{}_data = text_preprocessing(data=data, item = '{}', contraction_map=CONTRACTION_MAP, drop_digits=False, remove_stopwords=False, stemming=False).format(item, item), locals(), globals()" )
# 
# # Cleaning
# del data
# gc.collect()
# 
# """

"""##### *Clean flawed examples*

<hr>

- drop examples based on the threshold

# drop examples with an invalid ratio of length of text and headline
text_len = [len(t) for t in text_data]
head_len = [len(h) for h in headline_data]

print('Some statistics')

print('Average length of articles is {:.2f}.'.format(np.array(text_len).mean()))
print('Min = {:.0f}, Max = {:.0f}, Std = {:.2f}'.format(min(text_len), max(text_len), np.array(text_len).std()))

print('-----')

print('Average length of summaries is {:.2f}.'.format(np.array(head_len).mean()))
print('Min = {:.0f}, Max = {:.0f}, Std = {:.2f}'.format(min(head_len), max(head_len), np.array(head_len).std()))

max_examples = 150000
max_threshold = 0.75

# drop examples with an invalid ratio of length of text and headline
text_len = [len(t) for t in text_data]
head_len = [len(h) for h in headline_data]

ratio = [h/t for t, h in zip(text_len, head_len)]

problems1 = [problem for problem, r in enumerate(ratio) if (r > max_threshold)]
print(len(problems1))
text_data, headline_data = np.delete(text_data, problems1), np.delete(headline_data, problems1)
print("Number of examples after filtering: {:.0f}".format(text_data.shape[0]))

# drop too long articles (to avoid struggles with CUDA memory) and too short
text_len = [len(t) for t in text_data]

problems2 = [problem for problem, text_length in enumerate(text_len) if ((text_length > 200) | (text_length < 10) )]
print(len(problems2))
text_data, headline_data = np.delete(text_data, problems2), np.delete(headline_data, problems2)
print("Number of examples after filtering: {:.0f}".format(text_data.shape[0]))

# drop too pairs with too short/long summaries
head_len = [len(h) for h in headline_data]

problems3 = [problem for problem, headline_len in enumerate(head_len) if ( (headline_len > 75) | (headline_len < 2) )]
print(len(problems3))
text_data, headline_data = np.delete(text_data, problems3), np.delete(headline_data, problems3)
print("Number of examples after filtering: {:.0f}".format(text_data.shape[0]))

# some cleaning
del text_len, head_len, ratio, problems1, problems2, problems3
gc.collect()

# trim the data to have only a subset of the data for our project
try:
  text_data, headline_data = text_data[:max_examples], headline_data[:max_examples]
except:
  pass

print(text_data.shape, headline_data.shape)

**LOAD ALREADY PREPROCESSED DATA**
"""

text_data = np.load('../data/text_data.npy', allow_pickle = True)
headline_data = np.load('../data/headline_data.npy', allow_pickle = True)

print(text_data.shape, headline_data.shape)

"""*Print some statistics*"""

# drop examples with an invalid ratio of length of text and headline
text_len = [len(t) for t in text_data]
head_len = [len(h) for h in headline_data]

print('Some statistics')

print('Average length of articles is {:.2f}.'.format(np.array(text_len).mean()))
print('Min = {:.0f}, Max = {:.0f}, Std = {:.2f}'.format(min(text_len), max(text_len), np.array(text_len).std()))

print('-----')

print('Average length of summaries is {:.2f}.'.format(np.array(head_len).mean()))
print('Min = {:.0f}, Max = {:.0f}, Std = {:.2f}'.format(min(head_len), max(head_len), np.array(head_len).std()))

"""##### *Split data into train/val/test set*

<hr>

It's crucial to do this split in this step so that a dictionary that will be created for our model won't contain any words from validation/test set which are not presented in the training data.
"""

np.random.seed(222)

split = np.random.uniform(0, 1, size = text_data.shape[0])

# Train set
text_train, headline_train = text_data[split <= 0.9], headline_data[split <= 0.9]
# Validation set
text_val, headline_val = text_data[(split > 0.9) & (split <= 0.95)], headline_data[(split > 0.9) & (split <= 0.95)]
# Test set
text_test, headline_test = text_data[split > 0.95], headline_data[split > 0.95]

del text_data, headline_data
gc.collect()

"""*Print some statistics*"""

print('Average lengths of articles is {:.2f}'.format(np.array([len(text) for text in text_train]).mean()))

print('Average lengths of sumaries is {:.2f}'.format(np.array([len(text) for text in headline_train]).mean()))

"""##### *Sort dataset from the longest sequence to the shortest one*"""

def sort_data(text, headline):
  """
  """
  headline = np.array(
      [y for x,y in sorted(zip(text, headline), key = lambda pair: len(pair[0]), reverse = True)]
  )
  text = list(text)
  text.sort(key = lambda x: len(x), reverse = True)
  text = np.array(text)

  return text, headline

# Train set
text_train, headline_train = sort_data(text_train, headline_train)
# Validation set
text_val, headline_val = sort_data(text_val, headline_val)
# Test set
text_test, headline_test = sort_data(text_test, headline_test)

"""### **Prepare dictionary and embeddings**

##### *Create a dictionary and prepare a digestible representation of the data*
"""

class LangDict:
  """
  Source: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
  """
  def __init__(self):
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0: "sos", 1: "eos"}
    self.n_words = 2

  def add_article(self, article):
    for word in article:
      self.add_word(word)

  def add_word(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1

# Create dictionary based on the training data
text_dictionary = LangDict()
headline_dictionary = LangDict()

for article in text_train:
  text_dictionary.add_article(article)

for article in headline_train:
  headline_dictionary.add_article(article)

print("There are {:.0f} distinct words in the untrimmed text dictionary".format(len(text_dictionary.word2index.keys())))
print("There are {:.0f} distinct words in the untrimmed headline dictionary".format(len(headline_dictionary.word2index.keys())))

# Trim a dictionary to the words with at least 10 occurences within the text
text_min_count = 1
head_min_count = 2

## TEXT DICTIONARY
subset_words = [word for (word, count) in text_dictionary.word2count.items() if count >= text_min_count]
text_dictionary.word2index = {word: i for (word, i) in zip(subset_words, range(len(subset_words)))}
text_dictionary.index2word = {i: word for (word, i) in zip(subset_words, range(len(subset_words)))}
text_dictionary.word2count = {word: count for (word, count) in text_dictionary.word2count.items() if count >= text_min_count}

## HEADLINE DICTIONARY
subset_words = [word for (word, count) in headline_dictionary.word2count.items() if count >= head_min_count]
headline_dictionary.word2index = {word: i for (word, i) in zip(subset_words, range(len(subset_words)))}
headline_dictionary.index2word = {i: word for (word, i) in zip(subset_words, range(len(subset_words)))}
headline_dictionary.word2count = {word: count for (word, count) in headline_dictionary.word2count.items() if count >= head_min_count}

print("There are {:.0f} distinct words in the trimmed text dictionary, where only word with at least {:.0f} occurences are retained".format(len(text_dictionary.word2index.keys()), text_min_count))
print("There are {:.0f} distinct words in the trimmed headline dictionary, where only word with at least {:.0f} occurences are retained".format(len(headline_dictionary.word2index.keys()), head_min_count))
del text_min_count, head_min_count, subset_words

"""*Add pad token*"""

## TEXT DICTIONARY
pad_idx = max(list(text_dictionary.index2word.keys())) + 1

text_dictionary.word2index['<pad>'] = pad_idx
text_dictionary.index2word[pad_idx] = '<pad>'

print(len(text_dictionary.index2word.keys()))

## HEADLINE DICTIONARY
pad_idx = max(list(headline_dictionary.index2word.keys())) + 1

headline_dictionary.word2index['<pad>'] = pad_idx
headline_dictionary.index2word[pad_idx] = '<pad>'

print(len(headline_dictionary.index2word.keys()))

"""##### *Extract embedding vectors for words we need*"""

# Commented out IPython magic to ensure Python compatibility.
# """
# %%time
# pre_train_weight = extract_weight(text_dictionary)
# pre_train_weight = np.array(pre_train_weight, dtype = np.float32)
# np.save('../data/embedding.npy', pre_train_weight)
# 
# pre_train_weight_head = extract_weight(headline_dictionary)
# pre_train_weight_head = np.array(pre_train_weight_head, dtype = np.float32)
# np.save('../data/embedding_headline.npy', pre_train_weight_head)
# 
# del embed_dict
# gc.collect()
# 
# 
# """
# 
# pre_train_weight = np.load('../data/embedding.npy')
# pre_train_weight_head = np.load('../data/embedding_headline.npy')

"""### **Transform the data**"""

# Train set
text_train, text_lengths_train, headline_train, headline_lengths_train = data2PaddedArray(text_train, headline_train, {'text_dictionary': text_dictionary,
                                                                                                                       'headline_dictionary': headline_dictionary},
                                                                                          pre_train_weight)
# Validation set
text_val, text_lengths_val, headline_val, headline_lengths_val = data2PaddedArray(text_val, headline_val, {'text_dictionary': text_dictionary,
                                                                                                           'headline_dictionary': headline_dictionary},
                                                                                  pre_train_weight)
# Test set
text_test, text_lengths_test, headline_test, headline_lengths_test = data2PaddedArray(text_test, headline_test, {'text_dictionary': text_dictionary,
                                                                                                                 'headline_dictionary': headline_dictionary},
                                                                                       pre_train_weight)

"""# **3 Training**

## **3.1 Generator - Pretraining**

<hr>

**Description**

for model_size in model_sizes:
  # Model specification
  grid = {'max_epochs': 25,
          'batch_size': 32,
          'learning_rate': 3e-4,
          'clip': 10,
          'l2_reg': 1e-4,
          'model_name': "generator{:.0f}".format(model_size)
        }

  ##### model ######
  OUTPUT_DIM = len(headline_dictionary.index2word.keys())
  ENC_EMB_DIM = pre_train_weight.shape[1]
  ENC_HID_DIM = model_size
  DEC_HID_DIM = model_size

  enc_num_layers = 1 # number of layers in RNN
  dec_num_layers = 1 # number of layers in RNN

  ENC_DROPOUT = 0.1
  DEC_DROPOUT = 0.1

  # Initialization
  Generator = generator(model = _Seq2Seq, loss_function = nn.CrossEntropyLoss, optimiser = optim.Adam, l2_reg = grid['l2_reg'], batch_size = grid['batch_size'],
                      text_dictionary = text_dictionary, embeddings = pre_train_weight, max_epochs = grid['max_epochs'], learning_rate = grid['learning_rate'],
                      clip = grid['clip'], teacher_forcing_ratio = 1, OUTPUT_DIM = OUTPUT_DIM, ENC_HID_DIM = ENC_HID_DIM, ENC_EMB_DIM = ENC_EMB_DIM,
                      DEC_HID_DIM = DEC_HID_DIM, ENC_DROPOUT = ENC_DROPOUT, DEC_DROPOUT = DEC_DROPOUT, enc_num_layers = enc_num_layers, dec_num_layers = dec_num_layers,
                      device = device, model_name = grid['model_name'], push_to_repo = push_to_repo)
  
  # Load model if any
  Generator.load()

  # Prin model design and total number of parameters
  print(Generator.model)
  model_parameters = filter(lambda p: p.requires_grad, Generator.model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print("Model with RNN hidden size of {:.0f} has {:.0f} parameters in total.".format(model_size, params))

  # Run training
  Generator.train(X_train = text_train,
                y_train = headline_train,
                X_val = text_val,
                y_val = headline_val,
                X_train_lengths = text_lengths_train,
                y_train_lengths = headline_lengths_train,
                X_val_lengths = text_lengths_val,
                y_val_lengths = headline_lengths_val)

*Plot losses*
"""

fig, ax = plt.subplots(1, 3, figsize = (18, 5))

# Loss for 128
y = np.loadtxt('Results/generator128__train_loss.txt')
ax[0].plot(range(1, y.shape[0]+1), y, label = 'Train. loss')
y = np.loadtxt('Results/generator128__validation_loss.txt')
ax[0].plot(range(1, y.shape[0]+1), y, label = 'Val. loss')
ax[0].legend()
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Cross-Entropy loss')
ax[0].set_title(f'Model size = 128, Min validation loss = {y.min():.3f}')

# Loss for 256
y = np.loadtxt('Results/generator256__train_loss.txt')
ax[1].plot(range(1, y.shape[0]+1), y, label = 'Train. loss')
y = np.loadtxt('Results/generator256__validation_loss.txt')
ax[1].plot(range(1, y.shape[0]+1), y, label = 'Val. loss')
ax[1].legend()
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Cross-Entropy loss')
ax[1].set_title(f'Model size = 256, Min validation loss = {y.min():.3f}')

# Loss for 512
y = np.loadtxt('Results/generator512__train_loss.txt')
ax[2].plot(range(1, y.shape[0]+1), y, label = 'Train. loss')
y = np.loadtxt('Results/generator512__validation_loss.txt')
ax[2].plot(range(1, y.shape[0]+1), y, label = 'Val. loss')
ax[2].legend()
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Cross-Entropy loss')
ax[2].set_title(f'Model size = 512, Min validation loss = {y.min():.3f}')

plt.tight_layout()

"""## **3.2 Generator - Generating summaries**

<hr>

**Description:**

- Generate summaries
- Compute ROUGE metrics on training and validation set
- Generate training and validation data for discriminator

# setting
rouge = Rouge()
def rouge_get_scores(hyp, ref, n):
  try:
    return float(rouge.get_scores(hyp, ref)[0]['rouge-{}'.format(n)]['f'])
  except:
    return "drop"

pad_idx = headline_dictionary.word2index['<pad>']
eos_idx = headline_dictionary.word2index['eos']

for model_size in model_sizes:
 # Model specification
  grid = {'max_epochs': 25,
          'batch_size': 32,
          'learning_rate': 3e-4,
          'clip': 10,
          'l2_reg': 1e-4,
          'model_name': "generator{:.0f}".format(model_size)
        }

  ##### model ######
  OUTPUT_DIM = len(headline_dictionary.index2word.keys())
  ENC_EMB_DIM = pre_train_weight.shape[1]
  ENC_HID_DIM = model_size
  DEC_HID_DIM = model_size

  enc_num_layers = 1 # number of layers in RNN
  dec_num_layers = 1 # number of layers in RNN

  ENC_DROPOUT = 0.1
  DEC_DROPOUT = 0.1

  # Initialization
  Generator = generator(model = _Seq2Seq, loss_function = nn.CrossEntropyLoss, optimiser = optim.Adam, l2_reg = grid['l2_reg'], batch_size = grid['batch_size'],
                      text_dictionary = text_dictionary, embeddings = pre_train_weight, max_epochs = grid['max_epochs'], learning_rate = grid['learning_rate'],
                      clip = grid['clip'], teacher_forcing_ratio = 1, OUTPUT_DIM = OUTPUT_DIM, ENC_HID_DIM = ENC_HID_DIM, ENC_EMB_DIM = ENC_EMB_DIM,
                      DEC_HID_DIM = DEC_HID_DIM, ENC_DROPOUT = ENC_DROPOUT, DEC_DROPOUT = DEC_DROPOUT, enc_num_layers = enc_num_layers, dec_num_layers = dec_num_layers,
                      device = device, model_name = grid['model_name'], push_to_repo = push_to_repo)
  
  # Load model if any
  Generator.load()

  # Generate summaries for training data and save them
  hypotheses = Generator.generate_summaries(text_train, text_lengths_train, headline_train, headline_lengths_train)
  hypotheses = sum(
      [[' '.join([headline_dictionary.index2word[index] for index in batch[:, hypothesis] if (index != pad_idx) & (index != eos_idx)][1:]) for hypothesis in range(batch.shape[1])] for batch in hypotheses], []
  )
  references = [' '.join([headline_dictionary.index2word[index] for index in headline_train[:, sentence] if (index != pad_idx) & (index != eos_idx)][1:]) for sentence in range(headline_train.shape[1])]
  # trim
  lim = Generator.n_batches * grid['batch_size']
  references[:lim]

  rouge1 = [rouge_get_scores(hyp, ref, '1') for hyp, ref in zip(hypotheses, references)]
  rouge1 = np.array([x for x in rouge1 if x != 'drop']).mean()
  rouge2 = [rouge_get_scores(hyp, ref, '2') for hyp, ref in zip(hypotheses, references)]
  rouge2 = np.array([x for x in rouge2 if x != 'drop']).mean()
  rougel = [rouge_get_scores(hyp, ref, 'l') for hyp, ref in zip(hypotheses, references)]
  rougel = np.array([x for x in rougel if x != 'drop']).mean()
  
  # cleaning
  del hypotheses, references
  gc.collect()

  print('Model size = {:.0f}.'.format(model_size))
  print('ROUGE-1: {:.3f} on training data.'.format(100*np.array(rouge1)))
  print('ROUGE-2: {:.3f} on training data.'.format(100*np.array(rouge2)))
  print('ROUGE-l: {:.3f} on training data.'.format(100*np.array(rougel)))
  print('---------------')

  
  ROUGE = {'ROUGE-1': rouge1,
           'ROUGE-2': rouge2,
           'ROUGE-L': rougel}
  # save as json
  json_file = json.dumps(ROUGE)
  file = open('Results/ROUGE_{:.0f}_train.txt'.format(model_size), "w")
  file.write(json_file)
  file.close()

  # Generate summaries for training data and save them
  hypotheses = Generator.generate_summaries(text_val, text_lengths_val, headline_val, headline_lengths_val)
  hypotheses = sum(
      [[' '.join([headline_dictionary.index2word[index] for index in batch[:, hypothesis] if (index != pad_idx) & (index != eos_idx)][1:]) for hypothesis in range(batch.shape[1])] for batch in hypotheses], []
  )
  references = [' '.join([headline_dictionary.index2word[index] for index in headline_val[:, sentence] if (index != pad_idx) & (index != eos_idx)][1:]) for sentence in range(headline_val.shape[1])]
  # trim
  n_batches = len(references) // grid['batch_size']
  lim = n_batches * grid['batch_size']
  references[:lim]

  rouge1 = [rouge_get_scores(hyp, ref, '1') for hyp, ref in zip(hypotheses, references)]
  rouge1 = np.array([x for x in rouge1 if x != 'drop']).mean()
  rouge2 = [rouge_get_scores(hyp, ref, '2') for hyp, ref in zip(hypotheses, references)]
  rouge2 = np.array([x for x in rouge2 if x != 'drop']).mean()
  rougel = [rouge_get_scores(hyp, ref, 'l') for hyp, ref in zip(hypotheses, references)]
  rougel = np.array([x for x in rougel if x != 'drop']).mean()

  # cleaning
  del hypotheses, references
  gc.collect()
  
  print('Model size = {:.0f}.'.format(model_size))
  print('ROUGE-1: {:.3f} on validation data.'.format(100*np.array(rouge1)))
  print('ROUGE-2: {:.3f} on validation data.'.format(100*np.array(rouge2)))
  print('ROUGE-l: {:.3f} on validation data.'.format(100*np.array(rougel)))
  print('---------------')

  ROUGE = {'ROUGE-1': rouge1,
           'ROUGE-2': rouge2,
           'ROUGE-L': rougel}
  # save as json
  json_file = json.dumps(ROUGE)
  file = open('Results/ROUGE_{:.0f}_val.txt'.format(model_size), "w")
  file.write(json_file)
  file.close()

# Push everything to github
push_to_repo()

## **3.3 Generator - Generating datases for pretraining of discriminator**

<hr>

**Description:**

- We generate summaries as counterparts to real examples given by training and validation set to pretrain our generator.

- These summaries are generated using the best generator so far only. (we compare generators based upon ROUGE-1 metrics on validation set.

*Function for padding hypothesis*
"""

def padded_hypotheses(x, threshold, pad_idx):
  """
  :param x:
    type:
    description:
  :param threshold:
    type:
    description:
  :param pad_idx:
    type:
    description:

  :return x:
    type:
    description  
  """
  if x.shape[0] == threshold:
    return x
  else: 
    return np.r_[x, np.repeat(pad_idx, 32*(threshold - x.shape[0])).reshape(-1, 32)]

best_model_size = model_sizes[np.array([float(open('Results/ROUGE_{:.0f}_val.txt'.format(model_size), "r").read().split(', ')[0].split(': ')[1]) for model_size in model_sizes]).argmax()]

print("The model with hidden size of {:.0f} is the best performing model w.r.t. ROUGE-1 metric on validation set.".format(best_model_size))

"""# Model specification
grid = {'max_epochs': 25,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'clip': 10,
        'l2_reg': 1e-4,
        'model_name': "generator{:.0f}".format(best_model_size)
      }

##### model ######
OUTPUT_DIM = len(headline_dictionary.index2word.keys())
ENC_EMB_DIM = pre_train_weight.shape[1]
ENC_HID_DIM = best_model_size
DEC_HID_DIM = best_model_size

enc_num_layers = 1 # number of layers in RNN
dec_num_layers = 1 # number of layers in RNN

ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

# Initialization
Generator = generator(model = _Seq2Seq, loss_function = nn.CrossEntropyLoss, optimiser = optim.Adam, l2_reg = grid['l2_reg'], batch_size = grid['batch_size'],
                    text_dictionary = text_dictionary, embeddings = pre_train_weight, max_epochs = grid['max_epochs'], learning_rate = grid['learning_rate'],
                    clip = grid['clip'], teacher_forcing_ratio = 1, OUTPUT_DIM = OUTPUT_DIM, ENC_HID_DIM = ENC_HID_DIM, ENC_EMB_DIM = ENC_EMB_DIM,
                    DEC_HID_DIM = DEC_HID_DIM, ENC_DROPOUT = ENC_DROPOUT, DEC_DROPOUT = DEC_DROPOUT, enc_num_layers = enc_num_layers, dec_num_layers = dec_num_layers,
                    device = device, model_name = grid['model_name'], push_to_repo = push_to_repo)

# Load model if any
Generator.load()

# Generate summaries for training data
hypotheses = Generator.generate_summaries(text_train, text_lengths_train, headline_train, headline_lengths_train)
# Pad hypotheses
hypotheses = np.concatenate(
    [padded_hypotheses(hypothesis, 68, headline_dictionary.word2index['<pad>']) for hypothesis in hypotheses], axis = 1
)
# Correct the 'sos' symbol
hypotheses[0, :] = 0
# Concatenate real and fake summaries + transpose
real_fake_train = np.concatenate((headline_train, hypotheses), axis = 1)
real_fake_train = np.swapaxes(real_fake_train, 0, 1) # shape [n_examples, seq_len]
# add labels as the first column - 1 = Real, 0 = Generated
real_fake_train = np.c_[np.vstack((np.ones((headline_train.shape[1], 1)), np.zeros((hypotheses.shape[1], 1)))), real_fake_train]
# save
np.save('../data/real_fake_train.npy', real_fake_train)
del hypotheses

# Generate summaries for validation data
hypotheses = Generator.generate_summaries(text_val, text_lengths_val, headline_val, headline_lengths_val)
# Pad hypotheses
hypotheses = np.concatenate(
    [padded_hypotheses(hypothesis, headline_val.shape[0], headline_dictionary.word2index['<pad>']) for hypothesis in hypotheses], axis = 1
)
# Correct the 'sos' symbol
hypotheses[0, :] = 0
# Concatenate real and fake summaries + transpose
real_fake_val = np.concatenate((headline_val, hypotheses), axis = 1)
real_fake_val = np.swapaxes(real_fake_train, 0, 1) # shape [n_examples, seq_len]
# add labels as the first column - 1 = Real, 0 = Generated
real_fake_val = np.c_[np.vstack((np.ones((headline_val.shape[1], 1)), np.zeros((hypotheses.shape[1], 1)))), real_fake_val]
# reshuffle
np.random.shuffle(real_fake_val)
# save
np.save('../data/real_fake_val.npy', real_fake_val)

## **3.4 Discriminator - Pretraining**

<hr>

**Description:**
"""

# Load the data
real_fake_train = np.load('../data/real_fake_train.npy', allow_pickle = False)
real_fake_val = np.load('../data/real_fake_val.npy', allow_pickle = False)

# Split into X and y
X_train, y_train = torch.from_numpy(real_fake_train[:, 1:]).long(), torch.from_numpy(real_fake_train[:, 0]).long()
X_val, y_val = torch.from_numpy(real_fake_val[:, 1:]).long(), torch.from_numpy(real_fake_val[:, 0]).long()

best_val_loss = float('inf')
for n_kernels in [10, 20, 30, 50, 100]:
  for dropout in [0.0, 0.2, 0.3, 0.5]:
    param = {'max_epochs': 80,
            'learning_rate': 5e-4,
            'batch_size': 32,               
            'seq_len': 68,                   # length of your summary
            'embed_dim': 200,
            'drop_out': dropout,
            'kernel_num': n_kernels,                 # number of your feature map
            'in_channel': 1,                 # for text classification should be one
            # how many conv net are used in parallel in text classification
            'parallel_layer': 3,
            'model_name': 'n_{:.0f}_d_{:.0f}'.format(n_kernels, 10*dropout),
            'device':'cuda'}
    print('----------')
    print(f'Kernel filters = {n_kernels:.0f}, Dropout prob. = {dropout:.1f}')
    drt = Discriminator_utility(pre_train_weight_head,**param)
    drt.run_epochs(X_train,y_train,X_test = X_val, y_test = y_val)
    push_to_repo()
    # print accuracy on the validation data
    print("Kernel filters: {:.0f}, dropout: {:.1f} => Accuracy: {:.2f} %.".format(n_kernels, dropout, 100*drt.predict(X_val, y_val)))
    print('----------')
    if min(drt.val_losses) < best_val_loss:
      best_val_loss = min(drt.val_losses)
      best_n_kernels, best_dropout = n_kernels, dropout

print(f'The best performing model has {best_n_kernels:.0f} with drop. prob. {best_dropout:.1f} and performin loss of {best_val_loss:.3f} on validation data.')

"""*Depic losses for individual models*"""

y = np.loadtxt('Results/discriminator_n_10_d_3__validation_loss.txt')
plt.plot(range(1, y.shape[0]+1), y, label = 'Val. loss')

y = np.loadtxt('Results/discriminator_n_10_d_3__train_loss.txt')
plt.plot(range(1, y.shape[0]+1), y, label = 'Train. loss')

plt.legend()

plt.tight_layout()

fig, ax = plt.subplots(5, 4, figsize = (22, 17))

for n_kernels, i in zip([10, 20, 30, 50, 100], range(5)):
  for dropout, j in zip([0.0, 0.2, 0.3, 0.5], range(4)):
    try:
      y = np.loadtxt(f'Results/discriminator_n_{n_kernels:.0f}_d_{10*dropout:.0f}__train_loss.txt')
      ax[i, j].plot(range(1, y.shape[0]+1), y, label = 'Train. loss')
      y = np.loadtxt(f'Results/discriminator_n_{n_kernels:.0f}_d_{10*dropout:.0f}__validation_loss.txt')
      ax[i, j].plot(range(1, y.shape[0]+1), y, label = 'Val. loss')

      ax[i,j].set_xlabel('Epochs')
      ax[i,j].set_ylabel('Bin. Cross-Entropy')
      ax[i,j].set_title(f'Min. validation loss = {y.min():.3f}')
      ax[i,j].legend()
    except:
      pass

for a, prob in zip(ax[0], [0.0, 0.2, 0.3, 0.5]):
    a.annotate(f'Dropout = {prob:.1f}', xy=(0.5, 1.2), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

for a, filters in zip(ax[:,0], [10, 20, 30, 50, 100]):
    a.annotate(f'Kernels = {filters:.0f}', xy=(-0.2, 0.5), xytext=(-5, 0),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='right', va='center')

plt.tight_layout()



"""## **3.5 ADVERSARIAL TRAINING**

<hr>

**Description:**
"""