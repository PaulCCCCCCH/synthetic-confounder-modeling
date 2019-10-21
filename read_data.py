import pickle

f = open("data/embedding_matrix.pkl", "rb")
emb_mat = pickle.load(f)

f = open("data/vocab.pkl", "rb")
vocab = pickle.load(f)

