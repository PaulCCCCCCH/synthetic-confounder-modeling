from collections import Counter
from os import path, mkdir
import pickle
from nltk.util import ngrams as nltk_ngrams
from numpy.random import choice, normal, binomial, uniform
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("numtogen", help="Input number of sample sentences to generate", type=int, default=10)
parser.add_argument("grams", help="Specify the number n in n-gram", type=int, default=3)

parser.add_argument("--rebuild", action="store_true", default=False, help="Rebuild vocab and word effect")
parser.add_argument("--filter_n", type=int, help="Consider only the subset of vocabulary that appeared greater than or equal to n times", default=3)
parser.add_argument("--outdir", default="./out", help="Define output path")

args = parser.parse_args()

NGRAM = args.grams
NUM_TO_GEN = args.numtogen
EMB_DIM = 20

# Reading files
fs = open("data/s.txt", "rb")
fl = open("data/l.txt", "rb")
sentences = fs.readlines()
sentences = [s.lower() for s in sentences]
labels = list(map(int, fl.readlines()))
assert len(sentences) == len(labels)
sentence_count = len(sentences)
fs.close()
fl.close()

# Building vocabulary
if not path.exists("data/vocab.pkl") or args.rebuild:
    print("Building vocabulary")
    text_list = ("".join(sentences).split())
    counts = list(Counter(text_list).viewitems())
    counts.sort(key=lambda x: x[1], reverse=True)
    # Filter out least frequent words 
    counts_filtered = [(x[0], x[1]) for x in counts if x[1] >= args.filter_n]
    vocab_list = counts_filtered[:10000]
    vocab = dict(vocab_list)

    word_dict = dict()
    word_dict["<pad>"] = 0
    for i, word in enumerate(vocab.keys()):
        ip1 = i + 1 
        word_dict[word] = ip1
    #del word_dict["$t$"]
    #word_dict["$t$"] = 1

    f = open("data/vocab.pkl", "wb")
    pickle.dump(word_dict, f)
    f.close()

else:
    print("Loading vocabulary")
    f = open("data/vocab.pkl", "rb")
    word_dict = pickle.load(f)
    f.close()

vocab = word_dict
vocab_size = len(word_dict.keys())
total_words = sum(vocab.values())

embedding_matrix = np.array(uniform(-1,1,[vocab_size, EMB_DIM]))
embedding_matrix[0] = np.zeros(EMB_DIM)
embedding_matrix[1] = np.zeros(EMB_DIM)

f = open("data/embedding_matrix.pkl", "wb")
pickle.dump(embedding_matrix, f)
f.close()


if not path.exists("data/effect_list.pkl") or args.rebuild:
    print("Building effect dictionary for words")

    # Splitting sentences into 3 labels
    sentences_all_labels = ([], [], []) #neg, neu, pos
    for sent, label in zip(sentences, labels):
        label = label + 1
        sentences_all_labels[label].append(sent)

    sent_neg, sent_neu, sent_pos = sentences_all_labels

    # Calculating word effects using posterior probability by observing labels

    raw_effect_all_labels = [None, None, None]
    for i, ss in enumerate(sentences_all_labels):
        raw_effects = dict()
        counts = list(Counter("".join(ss).split()).viewitems())
        label_count = len(ss)
        for word, count in counts:
            posterior = float(count) * sentence_count / (label_count * total_words)
            ##### TODO: May need to fine tune the formula #####
            raw_effects[word] = np.log(posterior + 10) + normal()
            ################
        raw_effect_all_labels[i] = raw_effects

    raw_effect_list = [None for _ in range(vocab_size)]
    effect_list = [None for _ in range(vocab_size)]
    real = raw_effect_all_labels
    for word in vocab.keys():
        neg, neu, pos = (real[0].get(word), real[1].get(word), real[2].get(word))
        raw_effect_list[word_dict[word]] = (neg, neu, pos)
        ##### TODO: May need to fine tune the formula 
        ##### Also, use add-one or discount method to get rid of None's
        if neg is None and pos is None:
            effect = 0
        elif neg is None:
            effect = 5 + normal()
        elif pos is None:
            effect = -5 + normal()
        elif pos - neg >= 1:
            effect = 5 + normal()
        elif neg - pos >= 1:
            effect = -5 + normal()
        elif pos > neg:
            effect = 1 + normal()
        elif neg > pos:
            effect = -1 + normal()
        else:
            effect = 0
        effect_list[word_dict[word]] = effect
        ###############


    f = open("data/effect_list.pkl", "wb")
    pickle.dump(effect_list, f)
    f.close()

else:
    print("Loading effect list for words")
    f = open("data/effect_list.pkl", "rb")
    effect_list = pickle.load(f)
    f.close()


if not path.exists("data/{}grams.pkl".format(NGRAM)) or args.rebuild:
    print("Building {}-grams".format(NGRAM))

    ngrams = []
    for s in sentences:
        g = list(nltk_ngrams(s.split(),
                NGRAM,
                pad_left=True,
                left_pad_symbol="<s>",
                pad_right=True,
                right_pad_symbol="</s>"))

        ngrams.extend(g)

    # TODO: Filter out the least frequent words
    ngrams_count = dict(Counter(ngrams).viewitems())
    total_count = sum(ngrams_count.values())
    ngrams = dict([(g, float(ngrams_count[g]) / total_count) for g in ngrams_count.keys()])

    f = open("./data/{}grams.pkl".format(NGRAM), "wb")
    pickle.dump(ngrams, f)
    f.close()

else:
    print("Loading {}-grams".format(NGRAM))
    f = open("./data/{}grams.pkl".format(NGRAM), "rb")
    ngrams = pickle.load(f)
    f.close()

print("Generating samples")
samples = []
generated = 0
# for i in range(NUM_TO_GEN):
while generated < NUM_TO_GEN:
    cur_sentence = ["<s>"] * NGRAM
    while "</s>" not in cur_sentence:
        cur_window = cur_sentence[-NGRAM:]
        candidates = []
        probs = []
        for w in vocab:
            new_window = tuple(cur_window[1:] + [w])
            # TODO: Need to give those never appeared a chance
            if new_window in ngrams:
                candidates.append(w)
                probs.append(ngrams[new_window])

        if not candidates:
            break

        # In case probs were too long
        """
        if len(probs) > 32:
            temp = [(c, p) for c, p in sorted(zip(candidates[:32], probs[:32]), key=lambda t: t[1], reverse=True)]

            candidates = [c for c, _ in temp]
            probs = [p for _, p in temp]
        """

        probs = np.asarray(probs) / sum(probs)
        word = choice(candidates, p = probs)
        cur_sentence.append(word)

    cur_sentence = cur_sentence[NGRAM: ]
    for word in cur_sentence:
        # Accept the sample only when it has words with strong effects in it
        if abs(effect_list[word_dict[word]]) > 3:
            sample = dict()
            # Record the sentence as list of words
            sample["sentence"] = cur_sentence
            # Record effect vector of all the words in the sentence
            sample["effect"] = np.asarray([effect_list[word_dict[word]] for word in cur_sentence])
            # Record bag-of-words representation of the sentence
            bow = np.asarray([0 for _ in range(vocab_size)])
            for word in cur_sentence:
                bow[word_dict[word]] = 1
            sample["bow_repr"] = bow

            sample["sentence_ind"] = np.asarray([word_dict[word] for word in cur_sentence])

            r = np.dot(bow, effect_list) + normal()
            p = 1 / (1 + np.exp(-r))
            label = binomial(1, p)
            sample["label"] = label

            samples.append(sample)
            generated += 1
            if generated % 100 == 0:
                print("{} / {} generated".format(generated, NUM_TO_GEN))
            break

print("Showing first two examples")
print(samples[0])
print(samples[1])



print("Writing output files")
if not path.exists(args.outdir):
    mkdir(args.outdir)

f = open(args.outdir + "/samples.pkl", "wb")
pickle.dump(samples, f)
f.close()

f = open(args.outdir + "/samples.txt", "wb")
for i in samples:
    line = str(i["label"]) + " " + " ".join(i["sentence"]) + "\n"
    f.write(line)
f.close()

print("All done")
