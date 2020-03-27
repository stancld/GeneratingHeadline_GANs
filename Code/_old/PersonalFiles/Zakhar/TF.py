import sys

terms = {}


def update(tokens):
    for token in tokens:
        if token in terms:
            terms[token] += 1
        else:
            terms[token] = 1


# go over terms and update their tf
for line in sys.stdin:
    update(line.split())

# save and pass terms
f = open('tf.txt', 'w')

for term in list(terms):
    print(term)
    f.write(term + '\n')

f.close()
