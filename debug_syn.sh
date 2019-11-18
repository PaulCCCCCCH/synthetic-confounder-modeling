# Tests all passed
python main.py reg_attention debug_keyword_model_entropy_reg_attention \
        --reg_method entropy \
        --epoch 5 \
        --debug \
        --learning_rate 0.01

python main.py reg_attention debug_keyword_model_weight_reg_attention \
        --reg_method weight \
        --epoch 5 \
        --debug \
        --learning_rate 1.0

python main.py reg_attention debug_keyword_model_sparse_attention \
        --reg_method sparse \
        --epoch 5 \
        --debug \
        --learning_rate 1.0

python main.py baseline_mlp debug_keyword_model_mlp \
        --epoch 5\
        --debug

python main.py baseline_lstm debug_additive_lstm_entropy_reg_attention \
        --kwm_path ./models/debug_keyword_model_entropy_reg_attention \
        --epoch 5 \
        --debug \
        --learning_rate 1.0

python main.py baseline_lstm debug_additive_lstm_weight_reg_attention \
        --kwm_path ./models/debug_keyword_model_weight_reg_attention \
        --epoch 5 \
        --debug \
        --learning_rate 1.0

python main.py baseline_lstm debug_additive_lstm_sparse_attention \
        --kwm_path ./models/debug_keyword_model_sparse_attention \
        --epoch 5 \
        --debug \
        --learning_rate 1.0

python main.py adv_mlp debug_attention_adv_mlp \
        --epoch 5 \
        --debug \
        --learning_rate 0.1

python main.py hex_attention debug_attention_hex \
        --epoch 5 \
        --debug \
        --learning_rate 0.1


