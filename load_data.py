import pickle
import os
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def load_file(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def load_all_data(args):
    datapath = args.data_path
    if args.debug:
        sample_file = os.path.join("debug", "samples.pkl")
    else:
        sample_file = os.path.join(datapath, "samples.pkl")

    samples = load_file(sample_file)

    total_size = len(samples)
    train_size = int(total_size * 0.9)
    test_size = total_size - train_size

    all_x = pad_sequences(np.asarray([s['sentence_ind'] for s in samples]), maxlen=args.max_len, padding='post')
    all_y = np.asarray([s['label'] for s in samples])

    train_x = all_x[:train_size]
    train_y = all_y[:train_size]

    test_x = all_x[train_size: train_size + test_size]
    test_y = all_y[train_size: train_size + test_size]

    all_data = (
        train_x,
        train_y,
        test_x,
        test_y,
        load_file(os.path.join(datapath, "vocab.pkl")),
        load_file(os.path.join(datapath, "effect_list.pkl")),
        load_file(os.path.join(datapath, "embedding_matrix.pkl")),
    )

    return all_data





