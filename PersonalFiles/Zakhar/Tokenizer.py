import sys
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')


def tokenize(text):
    # make all character lowercase
    # text = text.lower()

    # remove newline characters
    text = text.rstrip()

    # replace unwanted chars (possibly close to the words) with 'space'
    for char in ['.', ',', ':', ';', '!', '?', '/', '…', '_', '|', '@']:
        text = text.replace(char, ' ')

    # keep only letters
    text = ''.join(filter(lambda x: x.isalpha() or x == ' ', text))

    # split words by the spaces
    text = text.split(' ')

    # remove 0 length words
    text = [stemmer.stem(word) for word in text if len(word) > 0]

    return text


# tokenize all input
for line in sys.stdin:
    print(' '.join(tokenize(line)))
