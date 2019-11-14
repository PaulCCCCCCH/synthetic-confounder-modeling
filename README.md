# Deconfounding Key-word Statistics 

## Dependencies:
- `tensorflow==1.15`
- `nltk`
- `keras`


## Usage
run `start.sh` or run `generate.py` and then `main.py`.
See parameters using `python generate.py -h` and `python main.py -h`

Generated dataset can be found in `./${outdir}/samples.pkl`

Visualization can be found in `./models/${name}/vis.html`

## Data Generation
Use `generate.py`. Parameters are defined as follows:
```
parser.add_argument("numtogen", help="Input number of sample sentences to generate", type=int, default=10)
parser.add_argument("grams", help="Specify the number n in n-gram", type=int, default=3)

parser.add_argument("--rebuild", action="store_true", default=False, help="Rebuild vocab and word effect")
parser.add_argument("--filter_n", type=int, help="Consider only the subset of vocabulary that appeared greater than or equal to n times", default=3)
parser.add_argument("--outdir", default="./data", help="Define output path")
```

## Training models
Use `main.py`. Parameters are defined as follows:
```
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
parser.add_argument("--learning_rate", type=float, default=0.1)
```


## Key variables
- `word_dict`: mapping from word(str) to index(int)
- `effect_list`: mapping from word index(int) to word effect(float) 
- `ngrams`: mapping from an ngram window(a tuple of strings) to a probability
- `samples`: a list of samples, each having the following fields:
    * `sentence`: the sentence as a list of strings
    * `sentence_ind`: the sentence as a list of word indices
    * `effect`: a list of float as effect for each word in the sentence
    * `bow_repr`: bag-of-word representation of the sentence (sum of each word as one-hot vectors)
    * `label`: 0 for negative, 1 for positive
  

## Vocabulary sizes:
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



