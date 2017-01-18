import json
from functools import reduce

from configs.constants import DATA_PATH, OUTPUT_VOCAB_PATH, EOS, INPUT_SEQUENCE_LENGTH
from utils.utils import tokenize

def get_pairs():
    with open(DATA_PATH, mode='r') as f:
        return json.load(f)


def prepare_training(pairs, w2v, vocab):
    def mapper(pair):
        name = map(lambda word: vocab.from_word(word), pair[0])
        name.append(vocab.from_word(EOS))
        description = encode_input(pair[1], w2v)
        return (name, description)
    pairs = map(mapper, pairs)
    # ignore long descriptions
    pairs = filter(lambda pair: len(pair[1]) <= INPUT_SEQUENCE_LENGTH and len(pair[0]) <= ANSWER_MAX_TOKEN_LENGTH, pairs)
    return pairs
