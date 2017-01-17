from gensim.utils import simple_preprocess

def tokenize(sentence):
    return simple_preprocess(sentence)
