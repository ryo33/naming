import logging
import json
import seq2seq
from seq2seq.models import SimpleSeq2Seq
from keras.models import load_model
import numpy as np

from word2vec.model import load_w2v_model
from utils.utils import tokenize
from configs.constants import (
        TRANSLATOR_MODEL_PATH, OUTPUT_VOCAB_PATH, EOS,
        TEST_FREQUENCY, INPUT_SEQUENCE_LENGTH,
        TOKEN_SIZE, TRAIN_BATCH_SIZE, ANSWER_MAX_TOKEN_LENGTH,
        HIDDEN_DIM, INPUT_LAYER_DEPTH, OUTPUT_LAYER_DEPTH)

logger = logging.getLogger(__name__)

def get_batches(pairs, batch_size):
    for i in range(0, len(l), batch_size):
        yield pairs[i : i + batch_size]


def get_training_batch(pairs, voc_size):
    batches_num = len(paris)
    for batch in get_batches(pairs, TRAIN_BATCH_SIZE):
        x = np.zeros((TRAIN_BATCH_SIZE, INPUT_SEQUENCE_LENGTH, TOKEN_SIZE), dtype=np.float)
        y = np.zeros((TRAIN_BATCH_SIZE, ANSWER_MAX_TOKEN_LENGTH, voc_size), dtype=np.bool)
        for batch_index, (name, description) in enumerate(batch):
            x[batch_index] = description
            for index, word in enumerate(name[:ANSWER_MAX_TOKEN_LENGTH]):
                y[batch_index, index, word] = 1
        yield x, y


def train_s2s_model(model, pairs, test_pairs, w2v, vocab):
    count = 0
    for x, y in get_batches(pairs, TRAIN_BATCH_SIZE):
        model.fit(x, y, batch_size=TRAIN_BATCH_SIZE, nb_epoch=1, verbose=1)
        if count % TEST_FREQUENCY == 0:
            for (name, description) in test_pairs:
                prediction = translate(description, s2s, w2v, vocab)
                logger.info('[%s](%s) -> [%s]' % (description, name, prediction))
            save_s2s_model(model)


def get_s2s_model(vocab_size):
    model = SimpleSeq2Seq(
            input_dim=TOKEN_SIZE,
            input_length=INPUT_SEQUENCE_LENGTH,
            hidden_dim=HIDDEN_DIM,
            output_dim=vocab_size,
            output_length=ANSWER_MAX_TOKEN_LENGTH,
            depth=(INPUT_LAYER_DEPTH, OUTPUT_LAYER_DEPTH))
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def save_s2s_model(model):
    model.save(TRANSLATOR_MODEL_PATH)


def load_s2s_model():
    return load_model(TRANSLATOR_MODEL_PATH)


class OutputVocab():
    def __init__(self, words):
        self.words = words
        self.size = len(words)
        self.index_to_word = dict(enumerate(words))
        self.word_to_index = {v: k for k, v in self.index_to_word.items()}

    def save(self):
        with open(OUTPUT_VOCAB_PATH, mode='w') as f:
            json.dump(self.words, f)

    def from_index(self, index):
        self.index_to_word[index]

    def from_word(self, word):
        self.word_to_index[word]


def load_vocab():
    with open(OUTPUT_VOCAB_PATH, mode='r') as f:
        return OutputVocab(json.load(f))


def get_vocab(pairs):
    words = set()
    for name, description in pairs:
        for word in name:
            words.add(word)
    words = list(words)
    words.sort()
    words.append(EOS)
    return OutputVocab(words)
