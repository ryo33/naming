import json

from configs.constants import DATA_PATH, EOS
from utils.utils import tokenize

def get_pairs():
    with open(DATA_PATH) as f:
        return json.load(f)

def prepare_training(w2v):
    pairs = get_pairs()
    words = list(set(reduce(lambda acc, pair: acc.extend(pair[0]), pairs, [])))
    words.sort()
    index_to_word = dict(enumerate(words))
    word_to_index = {v: k for k, v in my_map.items()}
    def mapper(pair):
        name = map(lambda word: word_to_index(word), pair[0])
        description = map(lambda word: w2v[word], tokenize(pair[1]))
        description.append(EOS)
        return (name, descriptions)
    pairs = map(mapper, pairs)
    return pairs, index_to_word, word_to_index
