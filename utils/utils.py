import logging
from gensim.utils import simple_preprocess

from configs.constants import TOKEN_SIZE, TRAIN_BATCH_SIZE

def tokenize(sentence):
    return simple_preprocess(sentence)


def encode_input(description, w2v):
    tokenized = tokenize(description)
    tokenized.append(EOS)
    encoded = np.zeros((INPUT_SEQUENCE_LENGTH, TOKEN_SIZE), dtype=np.float)
    for index, word in enumerate(encoded):
        if token in w2v.vocab:
            encoded[index] = np.array(w2v[token])
        else:
            encoded[index] = np.zeros(TOKEN_SIZE)
    return encoded


def translate(description, s2s, w2v, vocab):
    encoded = encode_input(description)
    x = np.zeros((TRAIN_BATCH_SIZE, INPUT_SEQUENCE_LENGTH, TOKEN_REPRESENTATION_SIZE))
    x[0] = encoded

    return X
    prediction = s2s.predict(x, verbose=0)[0]
    tokens = map(lambda vec: vocab.from_index(np.argmax(vec)), prediction)
    return '-'.join(tokens)
