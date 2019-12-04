import sys
import tensorflow as tf
import argparse
import models

class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


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


def get_args():
    reg_methods = ['none', 'weight', 'entropy', 'sparse']
    from model_utils import all_models
    model_types = all_models.keys()
    parser = argparse.ArgumentParser()

    parser.add_argument("modeltype", help="the type of models to choose from, choose from " + str(model_types))
    parser.add_argument("modelname", help="specify the name of the model", type=str)

    parser.add_argument("--test", action="store_true", default=False, help="Only test and produce visualisation")
    parser.add_argument("--task", type=str, default="synthetic", help="Choose from ['synthetic', 'snli', 'mnli'] ")
    parser.add_argument("--debug", action="store_true", default=False, help="Use debug dataset for quick debug runs")
    parser.add_argument("--lam", type=float, default=0.01, help="Coefficient of regularization term")
    parser.add_argument("--reg_method", type=str, default="none",
                        help="Specify regularization method for key-model weights. Default is 'none'. Choose from " + str(
                            reg_methods))
    parser.add_argument("--epochs", type=int, default=21, help="Specify epochs to train")
    parser.add_argument("--kwm_path", type=str, default="",
                        help="Specify a path to the pre-trained keyword model. Will only train a key-word model if left empty.")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum sentence length (excessive words are dropped)")
    parser.add_argument("--lstm_size", type=int, default=300, help="Size of lstm unit in current model.")
    parser.add_argument("--attention_size", type=int, default=128, help="Size of attention layer (if had one)")
    parser.add_argument("--embedding_dim", type=int, default=300, help="Dimension of embedding to use.")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Specify data directory (where inputs, effect list, vocabulary, etc. are )")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--keep_probs", type=float, default=0.5,
                        help="Keep probability of dropout layers. Set it to 1.0 to disable dropout.")
    parser.add_argument("--learning_rate", type=float, default=0.0004)
    parser.add_argument("--embedding_file", type=str, default="",
                        help="Specify path to the pre-trained embedding file, if had one.")
    parser.add_argument("--kwm_lstm_size", type=int, default=16, help="Only used in adversarial training models.")
    parser.add_argument("--step_size", type=int, default=50, help="Number of batches to run before running an evaluation")
    parser.add_argument("--vis_num", type=int, default=300, help="Defines the number of samples to include in the visualisation")
    parser.add_argument("--patience", type=int, default=10000, help="Max number of steps without improvements before an early stop")

    args = parser.parse_args()
    if args.modeltype not in model_types:
        raise NotImplementedError("Model type invalid")
    return args

