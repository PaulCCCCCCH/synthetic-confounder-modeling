import pickle
import os
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import collections
import random
import re


def load_file(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def load_all_data(args):
    """
    Load all data required to run the synthetic task
    """
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


LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
}

SNLI_FILE_MAP = {
    "train": "snli_1.0_train.jsonl",
    "dev": "snli_1.0_dev.jsonl",
    "test": "snli_1.0_test.jsonl"
}

MNLI_FILE_MAP = {
    "train": "multinli_0.9_train.jsonl",
    "dev_matched": "multinli_0.9_dev_matched.jsonl",
    "dev_mismatched": "multinli_0.9_dev_mismatched.jsonl"
}

PADDING = "<PAD>"
UNKNOWN = "<UNK>"

def sentences_to_padded_index_sequences(args, word_indices, dataset):
    """
    Annotate dataset with feature vectors. Adding right-sided padding.
    """
    result = []
    for example in dataset:
        for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
            example[sentence + '_index_sequence'] = np.zeros(args.max_len, dtype=np.int32)

            token_sequence = tokenize(example[sentence])
            padding = args.max_len - len(token_sequence)

            for i in range(args.max_len):
                if i >= len(token_sequence):
                    index = word_indices[PADDING]
                else:
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[UNKNOWN]
                example[sentence + '_index_sequence'][i] = index

        result.append([example['sentence1_binary_parse_index_sequence'],
                       example['sentence2_binary_parse_index_sequence']])

    return np.asarray(result)


# Currently not used
def _load_all_data_mnli(args):
    dev_matched = load_nli(args, "dev_matched", snli=False)
    dev_mismatched = load_nli(args, "dev_mismatched", snli=False)
    train = load_nli(args, "train", snli=False)

    word_dict = build_dictionary(train)
    word_embedding = load_embedding_rand(args, word_dict)

    train_x = sentences_to_padded_index_sequences(args, word_dict, train)
    train_y = np.asarray([LABEL_MAP[s['gold_label']] for s in train])
    dev_matched_x = sentences_to_padded_index_sequences(args, word_dict, dev_matched)
    dev_matched_y = np.asarray([LABEL_MAP[s['gold_label']] for s in dev_matched])
    dev_mismatched_x = sentences_to_padded_index_sequences(args, word_dict, dev_mismatched)
    dev_mismatched_y = np.asarray([LABEL_MAP[s['gold_label']] for s in dev_mismatched])

    return train_x, train_y, dev_matched_x, dev_matched_y, dev_mismatched_x, dev_mismatched_y, word_dict, word_embedding

def load_all_data_mnli(args, word_dict):
    dev_matched = load_nli(args, "dev_matched", snli=False)
    dev_mismatched = load_nli(args, "dev_mismatched", snli=False)

    dev_matched_x = sentences_to_padded_index_sequences(args, word_dict, dev_matched)
    dev_matched_y = np.asarray([LABEL_MAP[s['gold_label']] for s in dev_matched])
    dev_mismatched_x = sentences_to_padded_index_sequences(args, word_dict, dev_mismatched)
    dev_mismatched_y = np.asarray([LABEL_MAP[s['gold_label']] for s in dev_mismatched])

    return dev_matched_x, dev_matched_y, dev_mismatched_x, dev_mismatched_y

def load_all_data_snli(args):
    dev = load_nli(args, "dev", snli=True)
    test = load_nli(args, "test", snli=True)
    train = load_nli(args, "train", snli=True)

    word_dict = build_dictionary(train)
    word_embedding = load_embedding_rand(args, word_dict)

    train_x = sentences_to_padded_index_sequences(args, word_dict, train)
    train_y = np.asarray([LABEL_MAP[s['gold_label']] for s in train])
    dev_x = sentences_to_padded_index_sequences(args, word_dict, dev)
    dev_y = np.asarray([LABEL_MAP[s['gold_label']] for s in dev])
    test_x = sentences_to_padded_index_sequences(args, word_dict, test)
    test_y = np.asarray([LABEL_MAP[s['gold_label']] for s in test])

    return train_x, train_y, dev_x, dev_y, test_x, test_y, word_dict, word_embedding


def load_nli(args, set, snli=True):
    path = args.data_path
    file_map = SNLI_FILE_MAP if snli else MNLI_FILE_MAP
    data_file = os.path.join(path, file_map[set])
    data = []
    with open(data_file) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data


def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()


def build_dictionary(dataset):
    word_counter = collections.Counter()
    for example in dataset:
        word_counter.update(tokenize(example['sentence1_binary_parse']))
        word_counter.update(tokenize(example['sentence2_binary_parse']))

    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary

    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices


def load_embedding_rand(args, word_indices):
    n = len(word_indices)
    m = args.embedding_dim
    emb = np.empty((n, m), dtype=np.float32)

    emb[:, :] = np.random.normal(size=(n, m))

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0:2, :] = np.zeros((1, m), dtype="float32")

    with open(args.embedding_file, 'r') as f:
        for i, line in enumerate(f):

            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb

