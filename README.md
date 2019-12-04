# Deconfounding Key-word Statistics

## Dependencies:
- `tensorflow==1.15`
- `nltk`
- `keras`
- `wget`

## Running NLI tasks:
For examples, see `run_nli.sh`
Use `python main.py [modeltype] [modelname]` where `modeltype` must be one of the following:
```
    'reg_attention':    LSTM + regularized attention
    'adv_mlp':          adversarial model which uses LSTM as the prediction model and an MLP as the keyword model
    'hex_attention':    LSTM + attention + HEX
    'baseline_lstm':    LSTM + attention baseline
    'baseline_mlp':     MLP baseline
    'baseline_bilstm':  BiLSTM + baseline               # NLI task only
    'bilstm_attention': BiLSTM + attention baseline,    # NLI task only
    'baseline_esim':    ESIM baseline                   # NLI task only
```
and `modelname` is a unique identifier of the model. If a model with specified name exists, it will be restored.
To train separate models, use different names.

**Very important arguments**
- `--task`: Choose from ['snli' and 'mnli'] for nli tasks. They differ only on the training set they use.
Choose 'synthetic' for biased-signal capturing task.
- `--reg_method`: If an attention layer is involved in a model, it can be regularized by specifying one of
['none', 'weight', 'entropy', 'sparse'].
- `--kwm_path`: If wish to train an additive model with pre-trained keyword model, specify the path to the
keyword model you wish to use. Its weights will be frozen an parameters will be restored automatically. So, the
argument you input will only apply to the additive model. Doing this will not change the saved keyword model.
Leave it empty if not using additive training.
- `--data_path`: A repository containing 6 json files (both snli and mnli). File names must be the following:
```
    snli_1.0_train.jsonl,
    snli_1.0_dev.jsonl,
    snli_1.0_test.jsonl,
    multinli_1.0_train.jsonl,
    multinli_1.0_dev_matched.jsonl,
    multinli_1.0_dev_mismatched.jsonl
```
- `--embedding_file`: e.g. "./glove.840B.300d.txt". Required for nli tasks. If the embedding you use is not
300-dimensional, please specify its dimension using `--embedding_dim`.

To see the list of all arguments, see `get_args` in `utils.py` or run `main.py -h`

## Results
Saved models, logs and visualisations can be found in `./models/<modelname>`

## Steps of adding new models
1. Go to `models_nli.py`, create a new class that extends NLI model. This will give you the following variables:
- self.e_hypo: The input hypothesis of shape (batch_size, max_len, embedding_dim)
- self.e_prem: The input premise of shape (batch_size, max_len, embedding_dim)
- self.y_holder: The label of shape (batch_size). Note that you may want to reshape it using
`tf.one_hot(self.y_holder, depth=3)` when calculating the cost

For other defined variables, see `class Model` definition in `models.py` and `class NLIModel` in `models_nli.py`.


2. Define a `build_model` function. The following variables must be calculated:
- self.logits: raw output of a dense layer
- self.y: softmax of self.logits
- self.optimizer: e.g. `tf.train.GradientDescentOptimizer`, `tf.train.AdamOptimizer`...
- self.cost: e.g. `tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_holder, depth=3), logits=self.logits))`
- self.train_op: e.g. `self.optimizer.minimize(self.cost)`
- self.accuracy: e.g. `tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))`

and, if you would like to have a visualisation of an attention layer or other weights, you need to set the following
variables
- self.use_alphas: set it to True
- self.alphas_hypo: the alpha values produced by `attention_layer` function, or something equivalent
with shape (batch_size, max_len)
- self.alphas_prem: same as above

3. Go to `model_utils.py` and add `<modelname> : <classname>` pair to "all_models" dictionary.

4. It's done. You can call it by passing the `--modelname` to `main.py`.


## Running models on synthetic dataset
run `runner_synthetic.sh` or run `generate.py` and then `main.py`.
See parameters using `python generate.py -h` and `python main.py -h`

**Arguments for generating synthetic dataset**
Use `generate.py`. Parameters are defined as follows:
```
parser.add_argument("numtogen", help="Input number of sample sentences to generate", type=int, default=10)
parser.add_argument("grams", help="Specify the number n in n-gram", type=int, default=3)

parser.add_argument("--rebuild", action="store_true", default=False, help="Rebuild vocab and word effect")
parser.add_argument("--filter_n", type=int, help="Consider only the subset of vocabulary that appeared greater than or equal to n times", default=3)
parser.add_argument("--outdir", default="./data", help="Define output path")
```

**Key variables** in generation task:
- `word_dict`: mapping from word(str) to index(int)
- `effect_list`: mapping from word index(int) to word effect(float)
- `ngrams`: mapping from an ngram window(a tuple of strings) to a probability
- `samples`: a list of samples, each having the following fields:
    * `sentence`: the sentence as a list of strings
    * `sentence_ind`: the sentence as a list of word indices
    * `effect`: a list of float as effect for each word in the sentence
    * `bow_repr`: bag-of-word representation of the sentence (sum of each word as one-hot vectors)
    * `label`: 0 for negative, 1 for positive


**Vocabulary sizes if decide to filter**:
```
>>> len([0 for x in list(counts) if x[1] >= 2])
2488
>>> len([0 for x in list(counts) if x[1] >= 3])
1891
>>> len([0 for x in list(counts) if x[1] >= 4])
1486
>>> len([0 for x in list(counts) if x[1] >= 5])
1227
>>> len([0 for x in list(counts) if x[1] >= 6])
1060
>>> len([0 for x in list(counts) if x[1] >= 7])
926
>>> len([0 for x in list(counts) if x[1] >= 8])
828
>>> len([0 for x in list(counts) if x[1] >= 9])
758
>>> len([0 for x in list(counts) if x[1] >= 10])
692
```

## TODO:
- Put a reference at line 13 and 18 for HEX and ESIM
- Input of some models may not be masked (i.e. non-zero weights assigned to padding)