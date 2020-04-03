# Class Language Dictionary
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
