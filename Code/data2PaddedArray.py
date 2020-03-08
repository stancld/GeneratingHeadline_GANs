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

def data2PaddedArray(input, target, text_dictionary, embeddings):
    """
    :param input:
        type:
        description:
    :param target:
        type:
        description
        
    :return embedded_matrix:
        type: numpy.array
        description:
    :return input_seq_lengths:
        type: numpy.array
        description:
    :return padded_target:
        type: numpy.array
        description:
    :return target_seq_lengths:
        type: numpy.array
        description
    """
    # HELPER function
    def __word2index__(word, text_dictionary = text_dictionary, embeddings = embeddings):
      """
      :param word:
          type:
          description:
      :param text_dictionary:
          type:
          description:
      :param embeddings:
          type:
          description:
              
      :return word2index:
          type:
          description:  
      """
      try:
        word2index = text_dictionary.word2index[word]
      except:
        word2index = embeddings.shape[1] - 1
      return word2index
    
    # Create a vector of integers representing our text
    numericalVec_input = np.array(
        [[__word2index__(word) for word in sentence] for sentence in input]
        )
    numericalVec_target = np.array(
        [[__word2index__(word) for word in sentence] for sentence in target]
        )
    
    ### Convert the input data to embedded representation
    max_lengths = np.array([len(sentence) for sentence in input]).max()
    padded_input, input_seq_lengths = [], []
    for sentence in numericalVec_input:
        input_seq_lengths.append(len(sentence))
        if len(sentence) == max_lengths:
            sentence = np.array(sentence).reshape((1,-1))
        else:
            pad_idx = text_dictionary.word2index['<pad>']
            sentence = np.c_[np.array(sentence).reshape((1,-1)),
                             np.repeat(pad_idx, max_lengths - len(sentence)).reshape((1, -1))
                             ]
        padded_input.append(sentence)

    input_seq_lengths = np.array(input_seq_lengths, np.int)
    del numericalVec_input 

    ### Pad the target data
    max_lengths = np.array([len(sentence) for sentence in target]).max()
    padded_target, target_seq_lengths = [], []
    for sentence in numericalVec_target:
        target_seq_lengths.append(len(sentence))
        if len(sentence) == max_lengths:
            sentence = np.array(sentence).reshape((1,-1))
        else:
            pad_idx = text_dictionary.word2index['<pad>']
            sentence = np.c_[np.array(sentence).reshape((1,-1)),
                             np.repeat(pad_idx, max_lengths - len(sentence)).reshape((1, -1))
                             ]
        padded_target.append(sentence)
    
    target_seq_lengths = np.array(target_seq_lengths, np.int)
    
    del numericalVec_target
    
    return (np.array(padded_input, np.int32).squeeze(1).swapaxes(0,1), # => dims: [seq_length, n_examples, embedded_dim]
            input_seq_lengths,
            np.array(padded_target, np.int32).squeeze(1).swapaxes(0,1), # => dims: [seq_length, n_examples,]
            target_seq_lengths,
            )