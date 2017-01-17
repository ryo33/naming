from word2vec.model import train_model, save_model

def train():
    model = train_model()
    save_model(model)

if __name__ == "__main__":
    train()
