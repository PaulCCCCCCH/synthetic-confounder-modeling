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

python main.py baseline_lstm debug_snli_baseline_lstm \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py adv_mlp debug_snli_attention_adv_mlp \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py hex_attention debug_snli_attention_hex \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py baseline_bilstm debug_snli_bilstm \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py bilstm_attention debug_snli_bilstm_attention \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py bilstm_attention debug_snli_bilstm_attention \
        --reg_method weight \
        --epoch 5 \
        --debug \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py bilstm_attention debug_snli_bilstm_attention \
        --reg_method sparse\
        --epoch 5 \
        --debug \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py baseline_esim debug_snli_esim \
        --epoch 5 \
        --debug \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py baseline_esim debug_snli_additive_esim \
        --kwm_path ./models/debug_snli_keyword_model_entropy_reg_attention \
        --epoch 5 \
        --debug \
        --task snli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py reg_attention debug_mnli_keyword_model_entropy_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py reg_attention debug_mnli_keyword_model_weight_reg_attention \
        --reg_method weight\
        --epoch 5 \
        --debug \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py reg_attention debug_mnli_keyword_model_sparse_attention \
        --reg_method sparse \
        --epoch 5 \
        --debug \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py reg_attention debug_mnli_additive_entropy_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task mnli \
        --kwm_path ./models/debug_mnli_keyword_model_entropy_reg_attention \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py reg_attention debug_mnli_additive_weight_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task mnli \
        --kwm_path ./models/debug_mnli_keyword_model_weight_reg_attention \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py reg_attention debug_mnli_additive_sparse_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --kwm_path ./models/debug_mnli_keyword_model_sparse_attention \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py baseline_lstm debug_mnli_baseline_lstm \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py adv_mlp debug_mnli_attention_adv_mlp \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10

python main.py hex_attention debug_mnli_attention_hex \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py baseline_bilstm debug_mnli_bilstm \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py bilstm_attention debug_mnli_bilstm_attention \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py bilstm_attention debug_mnli_bilstm_attention \
        --reg_method weight \
        --epoch 5 \
        --debug \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py bilstm_attention debug_mnli_bilstm_attention \
        --reg_method sparse\
        --epoch 5 \
        --debug \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py baseline_esim debug_mnli_esim \
        --epoch 5 \
        --debug \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01

python main.py baseline_esim debug_mnli_additive_esim \
        --kwm_path ./models/debug_mnli_keyword_model_entropy_reg_attention \
        --epoch 5 \
        --debug \
        --task mnli \
        --learning_rate 0.01 \
        --data_path /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/ \
        --embedding_file /mnt/D/work/tensorflow/hex/stub_data/snli_1.0/glove.840B.300d.txt \
        --batch_size 10 \
        --lam 0.01
