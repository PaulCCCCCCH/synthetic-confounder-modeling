# Generate dataset of size 100000 using 3-grams, filter out words appeared less than 3 times
# python generate.py 100000 3 --rebuild --fil 3 --outdir ./data

# Train a keyword model using mlp
python main.py reg_attention keyword_model_entropy_reg_attention \
        --reg_method entropy \
        --epoch 20 \
        --learning_rate 0.01

python main.py reg_attention keyword_model_weight_reg_attention \
        --reg_method weight \
        --epoch 20 \
        --learning_rate 1.0

python main.py reg_attention keyword_model_sparse_attention \
        --reg_method sparse \
        --epoch 20 \
        --learning_rate 1.0

python main.py baseline_lstm additive_lstm_entropy_reg_attention \
        --kwm_path ./models/keyword_model_entropy_reg_attention \
        --epoch 20 \
        --learning_rate 1.0

python main.py baseline_lstm additive_lstm_weight_reg_attention \
        --kwm_path ./models/keyword_model_weight_reg_attention \
        --epoch 20 \
        --learning_rate 1.0

python main.py baseline_lstm additive_lstm_sparse_attention \
        --kwm_path ./models/keyword_model_sparse_attention \
        --epoch 20 \
        --learning_rate 1.0
