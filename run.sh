function download_data() {
  mkdir nli_data
  cd nli_data

  # Get Glove word embedding
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip
  unzip glove.840B.300d.zip
  rm glove.840B.300d.zip

  # Get SNLI dataset
  wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
  unzip snli_1.0.zip
  rm snli_1.0.zip
  mv ./snli_1.0/*.jsonl .
  rm -rf snli_1.0

  # Get MNLI dataset
  wget https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
  unzip multinli_1.0.zip
  rm multinli_1.0.zip
  mv ./multinli_1.0/*.jsonl .
  rm -rf multinli_1.0

  cd ..
}


[ ! -d "nli_data" ] && download_data

# Train an entropy attention keyword model on snli
python main.py reg_attention snli_keyword_model_entropy_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --task snli \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \
        --patience 10000

# Train an esim model with additive training on snli (using previous keyword model)
python main.py baseline_esim snli_additive_esim_reg_attention \
        --kwm_path ./models/snli_keyword_model_entropy_reg_attention \
        --epoch 5 \
        --task snli \
        --data_path ./nli_data/ \
        --embedding_file ./nli_data/glove.840B.300d.txt \
        --attention_size 128

# Train an esim baseline model on snli
python main.py baseline_esim snli_esim \
        --epoch 5 \
        --task snli \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \
        --attention_size 128 \

# Train a cbow model with additive training on snli (using previous keyword model)
python main.py baseline_cbow snli_additive_cbow_reg_attention\
        --epoch 5 \
        --task snli \
        --kwm_path ./models/snli_keyword_model_entropy_reg_attention \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \


# Train a cbow baseline model on snli
python main.py baseline_cbow snli_cbow \
        --epoch 5 \
        --task snli \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \


# Train an entropy attention keyword model on mnli
python main.py reg_attention mnli_keyword_model_entropy_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --task mnli \
        --learning_rate 0.01 \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \

# Train an esim model with additive training on mnli (using previous keyword model)
python main.py baseline_esim mnli_additive_esim_reg_attention \
        --kwm_path ./models/mnli_keyword_model_entropy_reg_attention \
        --epoch 5 \
        --task mnli \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \

# Train an esim baseline model on mnli
python main.py baseline_esim mnli_esim \
        --epoch 5 \
        --task mnli \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \


# Train a cbow model with additive training on mnli (using previous keyword model)
python main.py baseline_cbow mnli_additive_cbow_reg_attention\
        --epoch 5 \
        --task mnli \
        --kwm_path ./models/mnli_keyword_model_entropy_reg_attention \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \


# Train a cbow baseline model on mnli
python main.py baseline_cbow mnli_cbow \
        --epoch 5 \
        --task mnli \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \


