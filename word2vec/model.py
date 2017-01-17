import logging
from gensim.models import Word2Vec

from utils.utils import tokenize
from configs.constants import WORD2VEC_MODEL_PATH, DESCRIPTIONS_PATH, WORD_SIZE

def train_model():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    class Sentences():
        def __iter__(self):
            with open(DESCRIPTIONS_PATH) as f:
                for line in f:
                    yield tokenize(line)
    sentences = Sentences()
    return Word2Vec(sentences, size=WORD_SIZE)

def save_model(model):
    model.save(WORD2VEC_MODEL_PATH)

def load_model():
    return Word2Vec.load(WORD2VEC_MODEL_PATH)
