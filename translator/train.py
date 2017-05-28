import logging
import random
import json

from translator.pretrain import get_pairs, prepare_training
from translator.model import get_vocab, get_s2s_model, train_s2s_model, load_s2s_model, save_s2s_weights
from word2vec.model import load_w2v_model
from configs.constants import TEST_PAIRS_NUM

def train():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    w2v = load_w2v_model()
    pairs = get_pairs()
    vocab = get_vocab(pairs)
    vocab.save()

    random.shuffle(pairs)
    test_pairs = prepare_training(pairs[:TEST_PAIRS_NUM], w2v, vocab)
    train_pairs = prepare_training(pairs[TEST_PAIRS_NUM:], w2v, vocab)

    model = load_s2s_model()
    train_s2s_model(model, train_pairs, test_pairs, w2v, vocab)
    save_s2s_weights(model)

if __name__ == "__main__":
    train()
