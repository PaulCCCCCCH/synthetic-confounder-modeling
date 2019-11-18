# Test all passed

python main.py reg_attention debug_snli_keyword_model_entropy_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py reg_attention debug_snli_keyword_model_weight_reg_attention \
        --reg_method weight\
        --epoch 5 \
        --debug \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py reg_attention debug_snli_keyword_model_sparse_attention \
        --reg_method sparse \
        --epoch 5 \
        --debug \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py reg_attention debug_snli_additive_entropy_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task snli \
        --kwm_path ./models/debug_snli_keyword_model_entropy_reg_attention \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py reg_attention debug_snli_additive_weight_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task snli \
        --kwm_path ./models/debug_snli_keyword_model_weight_reg_attention \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py reg_attention debug_snli_additive_sparse_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --kwm_path ./models/debug_snli_keyword_model_sparse_attention \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10
