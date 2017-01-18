import logging
from word2vec.model import train_w2v_model, save_w2v_model

def train():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = train_w2v_model()
    save_w2v_model(model)

if __name__ == "__main__":
    train()
