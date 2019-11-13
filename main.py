import pickle
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import models
import numpy as np
import os
import vis_util
import argparse
import sys
import time
from load_data import load_all_data, load_file


def initialize_uninitialized_global_variables(sess):
    """
    Only initializes the variables of a TensorFlow session that were not
    already initialized.
    :param sess: the TensorFlow session
    :return:
    """
    # List all global variables
    global_vars = tf.global_variables()

    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
    return


all_models = {
    'reg_attention': models.RegAttention,
    'adv_mlp': models.LSTMPredModelWithMLPKeyWordModelAdvTrain,
    'hex_attention': models.LSTMPredModelWithRegAttentionKeyWordModelHEX,
    'baseline_lstm': models.LSTMPredModel,
    #'baseline_mlp': models.MLPPredModel  # Currently not working
}
model_types = all_models.keys()

reg_methods = ['none', 'weight', 'entropy', 'sparse']

parser = argparse.ArgumentParser()
parser.add_argument("modeltype", help="the type of models to choose from, choose from " + str(model_types))
parser.add_argument("modelname", help="specify the name of the model", type=str)

parser.add_argument("--test", action="store_true", default=False, help="Only test and produce visualisation")
parser.add_argument("--debug", action="store_true", default=False, help="Use debug dataset for quick debug runs")
parser.add_argument("--lam", type=float, default=0.01, help="Coefficient of regularization term")
parser.add_argument("--reg_method", type=str, default="none", help="Specify regularization method for key-model weights. Default is 'none'. Choose from " + str(reg_methods))
parser.add_argument("--epochs", type=int, default=21, help="Specify epochs to train")
parser.add_argument("--kwm_path", type=str, default="", help="Specify a path to the pre-trained keyword model. Will only train a key-word model if left empty." )
parser.add_argument("--max_len", type=int, default=30, help="Maximum sentence length (excessive words are dropped)")
parser.add_argument("--lstm_size", type=int, default=20, help="Size of lstm unit in current model.")
parser.add_argument("--embedding_dim", type=int, default=20, help="Dimension of embedding to use.")
parser.add_argument("--data_path", type=str, default="./data", help="Specify data directory (where inputs, effect list, vocabulary, etc. are )")
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--keep_probs", type=float, default=0.8, help="Keep probability of dropout layers. Set it to 1.0 to disable dropout.")

args = parser.parse_args()
if args.modeltype not in model_types:
    raise NotImplementedError("Model type invalid")

# Defining directories
ckpt_dir = os.path.join("models", args.modelname)
ckpt_file = os.path.join(ckpt_dir, args.modelname)
log_file = os.path.join(ckpt_dir, "log.txt")
arg_file = os.path.join(ckpt_dir, "args.pkl")

if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

if not os.path.exists(arg_file):
    with open(arg_file, "wb") as f:
        pickle.dump(args, f)
else:
    print("Restoring previous arguments from arg.pkl. See following summary args for details.")
    with open(arg_file, "rb") as f:
        args = pickle.load(f)

# Defining constants
datapath = args.data_path
max_len = args.max_len
batch_size = args.batch_size
lstm_size = args.lstm_size
num_epochs = args.epochs
embedding_dim = args.embedding_dim

# Redirect output to log file
log_fh = open(log_file, "a")
# sys.stdout = log_fh

# Print a summary of parameters
print("\n\n Started at " + str(time.ctime()))
print("Parameter summary")
print(args.__dict__)

#Loading data:

train_x, train_y, test_x, test_y, word_dict, effect_list, embedding_matrix = load_all_data(args)
vocab_size = embedding_matrix.shape[0]

print("Dataset building all done")

sess = tf.Session()
use_additive = False
if args.kwm_path != "":

    prev_arg_file = os.path.join(args.kwm_path, "args.pkl")
    prev_args = load_file(prev_arg_file)

    print("Loading key-word model with the following parameters: ")
    print(prev_args.__dict__)

    with tf.variable_scope(prev_args.modelname) as scope:
        key_word_model = models.get_model(prev_args, all_models, vocab_size, trainable=False)
    kwm_saver = tf.train.Saver()

    kwm_ckpt = os.path.join(args.kwm_path, prev_args.modelname)
    kwm_saver.restore(sess, kwm_ckpt)
    use_additive = True


with tf.variable_scope(args.modelname) as scope:
    pred_model = models.get_model(args, all_models, vocab_size)


saver = tf.train.Saver()

if use_additive:
    model = models.get_additive_model(pred_model, key_word_model)
else:
    model = pred_model

initialize_uninitialized_global_variables(sess)


print("Buidling the model. Model name: {}".format(args.modelname))


if args.test:
    saver.restore(sess, ckpt_file)
    print('Test accuracy = ', model.evaluate_accuracy(sess, test_x, test_y))
    print('Signal capturing score= ', model.evaluate_capturing(sess, test_x, test_y, effect_list))

else:
    sess.run(tf.assign(pred_model.embedding_w, embedding_matrix))

    if os.path.exists(ckpt_file+".meta"):
        print('Restoring Model')
        saver.restore(sess, ckpt_file)

    print('Training..')
    for i in range(num_epochs):
        epoch_loss, epoch_accuracy = model.train_for_epoch(sess, train_x, train_y)
        print(i,'loss: ', epoch_loss, 'acc: ', epoch_accuracy)
        #print('Train accuracy = ', model.evaluate_accuracy(sess, train_x, train_y))
        print('Test accuracy = ', model.evaluate_accuracy(sess, test_x, test_y))
        print('Signal capturing score= ', model.evaluate_capturing(sess, test_x, test_y, effect_list))
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    print("Saving the model")
    saver.save(sess, ckpt_file)
    print("Finished")


print("Producing visualization")
htmls = vis_util.knit(test_x, test_y, word_dict, effect_list, model, sess, 100)
f = open(os.path.join(ckpt_dir, "vis.html"), "wb")
for i in htmls:
    f.write(i)
f.close()

log_fh.close()


