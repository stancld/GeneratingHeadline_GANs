# ========================= RUN THIS IN TERMINAL IN CURRENT DIRECTORY ==========================
# ----------------------------------------------------------------------------------------------
# --- bzcat path_to_wikihow | python Tokenizer.py | python TF.py | python ReduceEmbedding.py ---
# ----------------------------------------------------------------------------------------------

import sys
from gensim.models import keyedvectors, Word2Vec

# initialise vars
path_to_original_vectors = '../../../../../../../../../../../Documents/Computing/DataSets/NLP/model_parts/Word2Vec/GoogleNews-vectors-negative300.bin'
dimensionality = 300
THRESHOLD_TF = 10


# get allowed words
allowed_words = set()

for line in sys.stdin:
    line = line.rstrip().split()
    if int(line[1]) > THRESHOLD_TF:
        allowed_words.add(line[0])

# load vectors
google_w2v = keyedvectors.KeyedVectors.load_word2vec_format(path_to_original_vectors,
                                                            binary=True)

# create sample Word2Vec model
new_w2v = Word2Vec([list(allowed_words)], min_count=1, size=dimensionality)

# iterate over words to include only the ones needed
for word in allowed_words:
    if word in google_w2v.wv.vocab:
        new_w2v.wv.vectors[new_w2v.wv.vocab[word].index] = google_w2v[word]

    if word.capitalize() in google_w2v.wv.vocab:
        new_w2v.wv.vectors[new_w2v.wv.vocab[word.capitalize()].index] = google_w2v[word.capitalize()]

# recalculate normalised vectors
new_w2v.wv.syn0norm = None
new_w2v.wv.init_sims()

# save model
new_w2v.wv.save_word2vec_format('reduced_vectors_' + str(dimensionality) + 'd.bin', binary=True)
