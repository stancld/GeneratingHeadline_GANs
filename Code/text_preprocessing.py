"""
GANs for Abstractive Text Summarization
Project for Statistical Natural Language Processing (COMP0087)
University College London

File: text_preprocessing.py

Collaborators:
    - Daniel Stancl
    - Dorota Jagnesakova
    - Guoliang HE
    - Zakhar Borok`
"""

import numpy as np
import pandas as pd
import re
import unicodedata
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords


def text_preprocessing(data, item, contraction_map,
                       drop_digits = False, remove_stopwords = False, stemming = False):
    """
    A function intended for an initial preprocessing of the raw text of 
    Wikihow dataset.
    
    :param data:                Pandas dataset of raw data with named columns.
    :param item:                Name of a column we want to preprocess.
        For WikiHow dataset two items are considered: 'text', 'headline'
    :param contraction_map:     Mapping dictionary of contraction terms. 
    :param drop_digits:         Boolean indicating if digits should be removed during preprocessing.
    :param remove_stopwords:    Boolean indicating if stopwords should be removed during preprocessing.
    :param stemming:            Boolean indicating if stemming should be executed during preprocessing.   
        
    :return text: Preprocesed and tokenized text returned in list. 
    """
    # Some check for the validity of parameters:
    assert type(data) == pd.core.frame.DataFrame, "Data must be stored in pandas dataframe."
    assert type(item) == str, "Item of the data must be indicated with string."
    assert type(contraction_map) == dict, "Contraction map must be specified in a dictionary"
    assert type(drop_digits) == bool, "A parameter drop_digits must be indicated with boolean."
    assert type(remove_stopwords) == bool, "A parameter remove_stopwords must be indicated with boolean."
    assert type(stemming) == bool, "A parameter stemming must be indicated with boolean."
    
    # remove extra commas in text/headline
    text = [re.sub(r'[.]+[\n]+[,]',".\n", article) for article in getattr(data, item)]    
    
    # strip spaces and commas etc at the beginning/end of sentences
    for pattern in [' ', ', ', ';\n', ';', '\n']:
        text = [sentence.strip(pattern) for sentence in text]
        
    # lower
    text = [sentence.lower() for sentence in text]
    
    # remove accented chracters
    text = [unicodedata.normalize('NFKD', sentence).encode('ascii', 'ignore').decode('utf-8', 'ignore') for sentence in text]
    
    # expand contractions
    contraction_patterns = re.compile(
        '({})'.format('|'.join(contraction_map.keys()))
        )
    def expand_match(contraction):
        """
        """
        return contraction.group(0)[0] + contraction_map.get(contraction.group(0))[1:]
    
    text = [contraction_patterns.sub(expand_match, sentence) for sentence in text]
    text = [re.sub("'", " ", sentence) for sentence in text]
    
    # add spaces before .!? and remove special characters
    pattern1 = r'([.!?])'
    text = [re.sub(pattern1, r" \1", sentence) for sentence in text]
    pattern2 = r'[^a-zA-z0-9.!?\s]+' if not drop_digits else r'[^a-zA-z.!?\s]+'
    text = [re.sub(pattern2, ' ', sentence) for sentence in text]
    
    # Stemming
    if stemming == True:
        stemmer = nltk.stem.porter.PorterStemmer()
        text = [
            [' '.join([stemmer.stem(word) for word in sentence.split()])
            for sentence in article] for article in text
            ]
    else:
        pass
    
    # remove stopwords
    if remove_stopwords == True:
        tokenizer = ToktokTokenizer()
        stopwords_list = stopwords.words('english')
        for special_case in ['no', 'not']:
            stopwords_list.remove(special_case)
        
        text = [
            [' '.join([token.strip() for token in tokenizer.tokenize(sentence) if token.strip() not in stopwords_list])
                for sentence in article]
            for article in text
            ]
    else:
        pass
    
    # split into individual tokens + add special symbols for a start and end of the sentence
    text = [('sos ' + sentence + ' eos').split() for sentence in text]
    
    # return numpy array
    return np.array(text)