import pickle

f = open("data/embedding_matrix.pkl", "rb")
emb_mat = pickle.load(f)

f = open("data/vocab.pkl", "rb")
vocab = pickle.load(f)

f = open("data/3grams.pkl", "rb")
ngram = pickle.load(f)

f = open("data/effect_list.pkl", "rb")
effect_list = pickle.load(f)

