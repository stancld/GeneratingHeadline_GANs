# -*- coding: utf-8 -*-
"""GANs for Abstractive Text Summarization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MBpliNJ1d0gMyPGIUr0z0EPrO65nNzJu

**Function keeping Colab running**

<hr>

1. Ctrl + Shift + I
2. Put the code below into the console

function ClickConnect(){
    console.log("Clicked on connect button"); 
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)

# **GANs for Abstractive Text Summarization**
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
reponame = 'GeneratingHeadlines_GANs'

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
# !rm -r GeneratingHeadlines_GANs
# !git clone https://github.com/stancld/GeneratingHeadlines_GANs.git
# 
# # Go to the NLP_Project folder
# %cd GeneratingHeadlines_GANs
# 
# # Config global user and add origin enabling us to execute push commands
# !git config --global user.email user_email
# !git remote rm origin
# !git remote add origin https://gansforlife:dankodorkamichaelzakgithub@github.com/stancld/GeneratingHeadlines_GANs.git

"""**Function push_to_repo**"""

def push_to_repo():
  """
  models_branch
  """
  !git remote rm origin
  !git remote add origin https://gansforlife:dankodorkamichaelzak@github.com/stancld/GeneratingHeadlines_GANs.git
  !git checkout master
  !git pull origin master
  !git branch models_branch
  !git checkout models_branch
  !git add .
  !git commit -m "model state update"
  !git checkout master
  !git merge models_branch
  !git push -u origin master

"""### **General stuff**

**Import essential libraries and load necessary conditionalities**
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import sys
import time
import gc

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

from gensim.models import Word2Vec

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

# code for the baseline model class _Seq2Seq()
run Code/Models/Attention_seq2seq.py

# code for the training class
run Code/Models/generator_training_class.py

"""### **Pretrained embeddings**

<hr>

**TODO:** *Put a comment which kind of embeddings we used. Add some references and so on*
"""

# Download and unzip GloVe embedding
#!wget http://nlp.stanford.edu/data/glove.6B.zip
#!unzip glove.6B.zip


# input your pre-train txt path and parse the data
path = '../data/glove.6B.100d.txt'
embed_dict = {}
with open(path,'r') as f:
  lines = f.readlines()
  for l in lines:
    w = l.split()[0]
    v = np.array(l.split()[1:]).astype('float')
    embed_dict[w] = v

embed_dict['@@_unknown_@@'] = np.random.random(100) # if we use 100 dimension embeddings

# remove all the unnecesary files
#!rm -rf glove.6B.zip
#!rm -rf glove.6B.50d.txt
#!rm -rf glove.6B.100d.txt
#!rm -rf glove.6B.200d.txt
#!rm -rf glove.6B.300d.txt

# check the length of the dictionary
len(embed_dict.keys())

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
      pre_train_weight = np.r_[pre_train_weight, np.zeros((1, 100))]
    
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

##### *Download and open data*
"""

data = pd.read_csv('../data/wikihowSep.csv',
                   error_bad_lines = False).astype(str)
print(data.shape)

"""##### *Clean flawed examples*

<hr>

- drop examples based on the threshold
"""

max_examples = 50000
max_threshold = 0.75

# drop examples with an invalid ratio of length of text and headline
text_len = [len(str(t)) for t in data.text]
head_len = [len(str(h)) for h in data.headline]

ratio = [h/t for t, h in zip(text_len, head_len)]

problems1 = [problem for problem, r in enumerate(ratio) if (r > max_threshold) & (problem in data.index)]
data = data.drop(index = problems1).reset_index().drop('index', axis = 1)

# drop too short and long articles (to avoid struggles with CUDA memory)
text_len = [len(str(t)) for t in data.text]

problems2 = [problem for problem, text_len in enumerate(ratio) if (text_len > 600) & (problem in data.index)]
data = data.drop(index = problems2)

# some cleaning
del text_len, head_len, ratio, problems1, problems2
gc.collect()

# trim the data to have only a subset of the data for our project
try:
  data = data[:max_examples]
except:
  pass

print(data.shape)

"""##### *Pre-process data*"""

# Commented out IPython magic to ensure Python compatibility.
# %time

for item in ['text', 'headline']:
  exec("""{}_data = text_preprocessing(data=data, item = '{}', contraction_map=CONTRACTION_MAP,
                                  drop_digits=False, remove_stopwords=False, stemming=False)""".format(item, item),
       locals(), globals()
  )

"""##### *Split data into train/val/test set*

<hr>

It's crucial to do this split in this step so that a dictionary that will be created for our model won't contain any words from validation/test set which are not presented in the training data.
"""

np.random.seed(222)

split = np.random.uniform(0, 1, size = data.shape[0])

# Train set
text_train, headline_train = text_data[split <= 0.9], headline_data[split <= 0.9]
# Validation set
text_val, headline_val = text_data[(split > 0.9) & (split <= 0.95)], headline_data[(split > 0.9) & (split <= 0.95)]
# Test set
text_test, headline_test = text_data[split > 0.95], headline_data[split > 0.95]

del data

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

for article in text_train:
  text_dictionary.add_article(article)

for article in headline_train:
  text_dictionary.add_article(article)

print("There are {:.0f} distinct words in the untrimmed dictionary".format(len(text_dictionary.word2index.keys())))

# Trim a dictionary to the words with at least 10 occurences within the text
min_count = 1
subset_words = [word for (word, count) in text_dictionary.word2count.items() if count >= min_count]
text_dictionary.word2index = {word: i for (word, i) in zip(subset_words, range(len(subset_words)))}
text_dictionary.index2word = {i: word for (word, i) in zip(subset_words, range(len(subset_words)))}
text_dictionary.word2count = {word: count for (word, count) in text_dictionary.word2count.items() if count >= min_count}

print("There are {:.0f} distinct words in the trimmed dictionary, where only word with at least {:.0f} occurences are retained".format(len(text_dictionary.word2index.keys()), min_count))
del min_count, subset_words

"""*Add pad token*"""

pad_idx = max(list(text_dictionary.index2word.keys())) + 1

text_dictionary.word2index['<pad>'] = pad_idx
text_dictionary.index2word[pad_idx] = '<pad>'

print(len(text_dictionary.index2word.keys()))

"""##### *Extract embedding vectors for words we need*"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# pre_train_weight = extract_weight(text_dictionary)
# pre_train_weight = np.array(pre_train_weight, dtype = np.float32)
# 
# del embed_dict
# gc.collect()

"""### **Transform the data**"""

# Train set
text_train, text_lengths_train, headline_train, headline_lengths_train = data2PaddedArray(text_train, headline_train, text_dictionary, pre_train_weight)
# Validation set
text_val, text_lengths_val, headline_val, headline_lengths_val = data2PaddedArray(text_val, headline_val, text_dictionary, pre_train_weight)
# Test set
text_test, text_lengths_test, headline_test, headline_lengths_test = data2PaddedArray(text_test, headline_test, text_dictionary, pre_train_weight)

"""# **3 Training**

## **3.1 Generator - Pretraining**

### **3.1.1 Version - 1**

<hr>

**Description**
"""

grid = {'max_epochs': 100,
        'batch_size': 96,
        'learning_rate': 5e-4,
        'clip': 10,
        'l2_reg': 1e-4,
        'model_name': "generator01"
      }

##### model ######
OUTPUT_DIM = len(text_dictionary.index2word.keys())
ENC_EMB_DIM = 100
ENC_HID_DIM = 128
DEC_HID_DIM = 128

ENC_DROPOUT = 0
DEC_DROPOUT = 0

Generator = generator(model = _Seq2Seq, loss_function = nn.CrossEntropyLoss, optimiser = optim.Adam, l2_reg = grid['l2_reg'], batch_size = grid['batch_size'],
                      text_dictionary = text_dictionary, embeddings = pre_train_weight, max_epochs = grid['max_epochs'], learning_rate = grid['learning_rate'],
                      clip = grid['clip'], teacher_forcing_ratio = 0.5, OUTPUT_DIM = OUTPUT_DIM, ENC_HID_DIM = ENC_HID_DIM, ENC_EMB_DIM = ENC_EMB_DIM,
                      DEC_HID_DIM = DEC_HID_DIM, ENC_DROPOUT = ENC_DROPOUT, DEC_DROPOUT = DEC_DROPOUT, device = device, model_name = grid['model_name'],
                      push_to_repo = push_to_repo)

Generator.load()

Generator.train(X_train = text_train,
                y_train = headline_train,
                X_val = text_val,
                y_val = headline_val,
                X_train_lengths = text_lengths_train,
                y_train_lengths = headline_lengths_train,
                X_val_lengths = text_lengths_val,
                y_val_lengths = headline_lengths_val)

Generator.model









