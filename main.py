import pickle
import os
import sys
import time
import utils
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

args = utils.get_args()
if args.task == "synthetic":
    import runner_synthetic as runner
elif args.task == "snli":
    args.embedding_dim = 300
    import runner_nli as runner
elif args.task == "mnli":
    raise NotImplementedError

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
logger = utils.Logger(log_file)
sys.stdout = logger

# Print a summary of parameters
print("\n\n Started at " + str(time.ctime()))
print("Parameter summary")
print(args.__dict__)

#Loading data:

################################Start runner here ############################
runner.run(args, ckpt_dir, ckpt_file)


