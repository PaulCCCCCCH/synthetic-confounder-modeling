# python generate.py 100000 3 --rebuild --fil 3
python main.py reg_attention debug_kwm --reg_method entropy --debug --epoch 10
python main.py baseline_lstm additive --kwm_path ./models/debug_kwm --debug
