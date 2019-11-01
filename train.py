import pickle
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import models
import numpy as np
import os
import vis_util
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("modelname", help="specify the name of the model", type=str)

parser.add_argument("--test", action="store_true", default=False, help="Only test and produce visualisation")
parser.add_argument("--debug", action="store_true", default=False, help="Use debug dataset for quick debug runs")
parser.add_argument("--lam", type=float, default=0, help="Coefficient of regularization term")
parser.add_argument("--reg_method", type=str, default="none", help="Specify regularization method. Choose from ['none', 'weight', 'entropy', 'sparse']. Default is 'none'")
parser.add_argument("--epochs", type=int, default=21, help="Specify epochs to train")

args = parser.parse_args()

# Defining constants
max_len = 30
batch_size = 10
lstm_size = 20
num_epochs = args.epochs
embedding_dim = 20

#Loading data:
print("Loading vocabulary")
f = open("data/vocab.pkl", "rb")
word_dict = pickle.load(f)
f.close()

if args.debug:
    print("Loading debug dataset")
    f = open("out_debug/samples.pkl", "rb")
    samples = pickle.load(f)
    f.close()
else:
    print("Loading dataset")
    f = open("out/samples.pkl", "rb")
    samples = pickle.load(f)
    f.close()

total_size = len(samples)
train_size = int(total_size * 0.9)
test_size = total_size - train_size


f = open('data/effect_list.pkl', 'rb') 
effect_list = pickle.load(f)
f.close()

f = open('data/embedding_matrix.pkl', 'rb') 
embedding_matrix = pickle.load(f)
f.close()

all_x = pad_sequences(np.asarray([s['sentence_ind'] for s in samples]), maxlen=30, padding='post')
all_y = np.asarray([s['label'] for s in samples])

train_x = all_x[:train_size]
train_y = all_y[:train_size]

test_x = all_x[train_size: train_size+test_size]
test_y = all_y[train_size: train_size+test_size]

sess = tf.Session()
print("Buidling the model. Model name: {}".format(args.modelname))

if args.reg_method == 'sparse':
    init = models.SentimentModelWithSparseAttention
else:
    init = models.SentimentModelWithRegAttention

model = init(batch_size=batch_size,
                       lstm_size = lstm_size,
                       max_len = max_len,
                       keep_probs=0.8,
                       embeddings_dim=embedding_matrix.shape[1], vocab_size=embedding_matrix.shape[0],
                       is_train=True,
                       reg=args.reg_method,
                       lam=args.lam)


sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
ckpt_dir = "./models/" + args.modelname + "/"
ckpt_file = ckpt_dir + args.modelname
if args.test:
    saver.restore(sess, ckpt_file)
    print('Test accuracy = ', model.evaluate_accuracy(sess, test_x, test_y))
    print('Signal capturing score= ', model.evaluate_capturing(sess, test_x, test_y, effect_list))

else:
    sess.run(tf.assign(model.embedding_w, embedding_matrix))

    if os.path.exists(ckpt_file+".meta"):
        print('Restoring Model')
        saver.restore(sess, ckpt_file)

    print('Training..')
    for i in range(num_epochs):
        epoch_loss, epoch_accuracy = model.train_for_epoch(sess, train_x, train_y)
        print(i,'loss: ', epoch_loss, 'acc: ', epoch_accuracy)
        #print('Train accuracy = ', model.evaluate_accuracy(sess, train_x, train_y))
        print('Test accuracy = ', model.evaluate_accuracy(sess, test_x, test_y))
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    print("Saving the model")
    saver.save(sess, ckpt_file)
    print("Finished")


print("Producing visualization")
htmls = vis_util.knit(test_x, test_y, word_dict, effect_list, model, sess, 100)
f = open(ckpt_dir + "vis.html", "wb")
for i in htmls:
    f.write(i)
f.close()



