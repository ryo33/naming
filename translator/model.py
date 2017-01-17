import logging
from keras.models import load_model as keras_load_model
import numpy as np

from utils.utils import tokenize
from configs.constants import TRANSLATOR_MODEL_PATH, WORD_SIZE

def get_training_batch(pairs, voc_size):
    batches_num = len(paris)
    batch_size = 1
    for batch in get_batches(pairs, batch_size):
        x = np.zeros((batches_num, ANSWER_MAX_WORD_LENGTH, voc_size), dtype=np.bool)
        y = np.zeros((batches_num, name.length, WORD_SIZE), dtype=np.float)
        for pair_index, (name, description) in enumerate(pairs):
            for index, word in enumerate(sents_batch[s_index][:INPUT_SEQUENCE_LENGTH]):
                x[pair_index, index] = get_word_vector(word, w2v)

            for index, word in enumerate(sents_batch[s_index + 1][:ANSWER_MAX_TOKEN_LENGTH]):
                y[pair_index, index, word_to_index[word]] = 1

       yield x, y

def train_model():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def save_model(model):
    model.save(TRANSLATOR_MODEL_PATH)

def load_model():
    return keras_load_model(TRANSLATOR_MODEL_PATH)
