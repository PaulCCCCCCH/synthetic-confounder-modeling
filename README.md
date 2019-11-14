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
- Word effects should be calculated in a better way



