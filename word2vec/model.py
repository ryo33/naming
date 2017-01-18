from gensim.models import Word2Vec

from utils.utils import tokenize
from configs.constants import WORD2VEC_MODEL_PATH, DESCRIPTIONS_PATH, TOKEN_SIZE

def train_w2v_model():
    class Sentences():
        def __iter__(self):
            with open(DESCRIPTIONS_PATH, mode='r') as f:
                for line in f:
                    yield tokenize(line)
    sentences = Sentences()
    return Word2Vec(sentences, size=TOKEN_SIZE)

def save_w2v_model(model):
    model.save(WORD2VEC_MODEL_PATH)

def load_w2v_model():
    return Word2Vec.load(WORD2VEC_MODEL_PATH)
