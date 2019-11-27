# Train an entropy attention keyword model on snli
python main.py reg_attention snli_keyword_model_entropy_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --task snli \
        --learning_rate 0.01 \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \
        --batch_size 10

# Train an esim model with additive training on snli (using previous keyword model)
python main.py baseline_esim snli_additive_esim_reg_attention \
        --kwm_path ./models/snli_keyword_model_entropy_reg_attention \
        --epoch 5 \
        --task snli \
        --learning_rate 0.01 \
        --data_path ./nli_data/ \
        --embedding_file /nli_data/glove.840B.300d.txt \
        --batch_size 10 \
        --attention_size 128

# Train an esim baseline model on snli
python main.py baseline_esim snli_esim \
        --epoch 5 \
        --task snli \
        --learning_rate 0.01 \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \
        --batch_size 10 \
        --attention_size 128

# Train an entropy attention keyword model on mnli
python main.py reg_attention mnli_keyword_model_entropy_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --task mnli \
        --learning_rate 0.01 \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \
        --batch_size 10

# Train an esim model with additive training on mnli (using previous keyword model)
python main.py baseline_esim snli_additive_esim_reg_attention \
        --kwm_path ./models/mnli_keyword_model_entropy_reg_attention \
        --epoch 5 \
        --task mnli \
        --learning_rate 0.01 \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \
        --batch_size 10 \
        --attention_size 128

# Train an esim baseline model on snli
python main.py baseline_esim snli_esim \
        --epoch 5 \
        --task snli \
        --learning_rate 0.01 \
        --data_path ./nli_data \
        --embedding_file ./nli_data/glove.840B.300d.txt \
        --batch_size 10 \
        --attention_size 128
# Train a keyword model using weight-regularized attention
#python main.py reg_attention snli_keyword_model_weight_reg_attention \
#        --reg_method weight\
#        --epoch 5 \
#        --task snli \
#        --learning_rate 0.01 \
#        --data_path ./nli_data \
#        --embedding_file ./nli_data/glove.840B.300d.txt \
#        --batch_size 10
#

# Train a keyword model using sparse attention
#python main.py reg_attention snli_keyword_model_sparse_attention \
#        --reg_method sparse \
#        --epoch 5 \
#        --task snli \
#        --learning_rate 0.01 \
#        --data_path ./nli_data \
#        --embedding_file ./nli_data/glove.840B.300d.txt \
#        --batch_size 10
#

# Train a keyword model using entropy-regularized attention
#python main.py reg_attention snli_additive_entropy_reg_attention \
#        --reg_method entropy \
#        --epoch 5 \
#        --task snli \
#        --kwm_path ./models/snli_keyword_model_entropy_reg_attention \
#        --learning_rate 0.01 \
#        --data_path ./nli_data \
#        --embedding_file ./nli_data/glove.840B.300d.txt \
#        --batch_size 10
#

