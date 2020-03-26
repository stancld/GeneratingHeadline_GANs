"""
GANs for Abstractive Text Summarization
Project for Statistical Natural Language Processing (COMP0087)
University College London

File: data2PaddedArray.py

Description of our model:

Collaborators:
    - Daniel Stancl
    - Dorota Jagnesakova
    - Guoliang HE
    - Zakhar Borok`
"""
import numpy as np

def data2PaddedArray(input, target, text_dictionary:dict, embeddings):
    """
    :param input:
        type: Numpy.Object: [n_examples]
        description: Sequences of input articles
    :param target:
        type: Numby.Object: [n_examples]
        description Sequences of tartget summaries
    :param text_dictionary:
        type: Dictionary
        description: Dictionary containing text_dictionary intended for input articles and headline dictionary used for target summaries
    ##
    :param embeddings: !! This parameter is to left here unintentionally from some preliminary versions. Desired to be deleted if some further updates are done in future.
    ##
         
    :return padded_input:
        type: Numpy.Array: [seq_Length, n_examples]
        description: Padded sequences of input articles
    :return input_seq_lengths:
        type: Numpy.Array: [n_examples]
        description: An array containing lengths of input articles
    :return padded_target:
        type: Numpy.Array: [seq_Length, n_examples]
        description: Padded sequences of target summaries
    :return target_seq_lengths:
        type: Numpy.Arrat: [n_examples]
        description: An array containing lengths of target summaries
    """
    # HELPER function
    def __word2index__(word, text_dictionary, embeddings = embeddings):
      """
      :param word:
          type: String
          description: A single word distilled from the input/target text
      :param text_dictionary:
          type: Dictionary
          description: A dictionary containing two other dictionary - text_dictionary intended for input artciles and headline_dictionary for targer summaries
      ##
      :param embeddings: !! This parameter is to left here unintentionally from some preliminary versions. Desired to be deleted if some further updates are done in future.
      ##
              
      :return word2index: 
          type: Integer
          description: Index to a given word (or unknown word) 
      """
      try:
        word2index = text_dictionary.word2index[word]
      except:
        word2index = embeddings.shape[1] - 1
      return word2index
    
    # Create a vector of integers representing our text
    numericalVec_input = np.array(
        [[__word2index__(word, text_dictionary = text_dictionary['text_dictionary']) for word in sentence] for sentence in input]
        )
    numericalVec_target = np.array(
        [[__word2index__(word, text_dictionary = text_dictionary['headline_dictionary']) for word in sentence] for sentence in target]
        )
    
    ### Pad the input articles
    # Derive the length of the longest input article
    max_lengths = np.array([len(sentence) for sentence in input]).max()
    # create an empty list for storing padded input articles and their original lengths
    padded_input, input_seq_lengths = [], []
    
    # Pad and store article by article from the input
    for sentence in numericalVec_input:
        input_seq_lengths.append(len(sentence))
        if len(sentence) == max_lengths:
            sentence = np.array(sentence).reshape((1,-1))
        else:
            pad_idx = text_dictionary['text_dictionary'].word2index['<pad>']
            sentence = np.c_[np.array(sentence).reshape((1,-1)),
                             np.repeat(pad_idx, max_lengths - len(sentence)).reshape((1, -1))
                             ]
        padded_input.append(sentence)
    # Put input_seq_lengths to Numpy array.
    input_seq_lengths = np.array(input_seq_lengths, np.int)
    del numericalVec_input 

    ### Pad the target data
    # Derive the length of the longest target summaries
    max_lengths = np.array([len(sentence) for sentence in target]).max()
    # create an empty list for storing padded target summaries and their original lengths
    padded_target, target_seq_lengths = [], []
    
    # Pad and store summary by summary from the target
    for sentence in numericalVec_target:
        target_seq_lengths.append(len(sentence))
        if len(sentence) == max_lengths:
            sentence = np.array(sentence).reshape((1,-1))
        else:
            pad_idx = text_dictionary['headline_dictionary'].word2index['<pad>']
            sentence = np.c_[np.array(sentence).reshape((1,-1)),
                             np.repeat(pad_idx, max_lengths - len(sentence)).reshape((1, -1))
                             ]
        padded_target.append(sentence)
    # Put target_seq_lengths to Numpy array.
    target_seq_lengths = np.array(target_seq_lengths, np.int)
    del numericalVec_target
    
    # return
    return (np.array(padded_input, np.int32).squeeze(1).swapaxes(0,1), # => dims: [seq_length, n_examples]
            input_seq_lengths,
            np.array(padded_target, np.int32).squeeze(1).swapaxes(0,1), # => dims: [seq_length, n_examples]
            target_seq_lengths,
            )