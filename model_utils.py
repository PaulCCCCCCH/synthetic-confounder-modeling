import tensorflow as tf

# Input must be of shape (Batch, TimeStep, HiddenSize)

all_models = {
    'reg_attention':    'models.RegAttention',
    'adv_mlp':          'models.LSTMPredModelWithMLPKeyWordModelAdvTrain',
    'hex_attention':    'models.LSTMPredModelWithRegAttentionKeyWordModelHEX',
    'baseline_lstm':    'models.LSTMPredModel',
    'baseline_mlp':     'models.MLPPredModel',
    'baseline_bilstm':  'models.BiLSTMPredModel',           # NLI task only
    'bilstm_attention': 'models.BiLSTMAttentionPredModel',  # NLI task only
    'baseline_esim':    'models.ESIMPredModel'              # NLI task only
}


def attention_layer(attention_size, inputs, name, sparse=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    w_omega = tf.get_variable('w_omega_'+name, initializer=tf.random_normal([int(inputs.shape[-1]), attention_size]))
    b_omega = tf.get_variable('b_omega_'+name, initializer=tf.random_normal([attention_size]))
    u_omega = tf.get_variable('u_omega_'+name, initializer=tf.random_normal([attention_size]))

    value = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(value, u_omega, axes=1, name='vu_'+name)
    if sparse:
        print("Using sparsemax attention")
        alphas = tf.contrib.sparsemax.sparsemax(vu, name='alphas_'+name)
    else:
        alphas = tf.nn.softmax(vu, name='alphas_'+name)
    last_output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return last_output, alphas


def dense_layer(inputs, out_dim, name, activation=None):
    w = tf.get_variable("w_" + name, shape=[inputs.shape[-1], out_dim], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable("b_" + name, shape=[out_dim], dtype=tf.float32)

    layer = tf.matmul(inputs, w) + b

    act = {
        "sigmoid": tf.nn.sigmoid,
        "relu": tf.nn.relu,
        "tanh": tf.nn.tanh,
    }
    if activation is not None:
        layer = act[activation](layer)
    return layer


def lstm_layer(inputs, lstm_size, batch_size, seq_len, scope=""):
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    init_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=lstm,
                                                 inputs=inputs,
                                                 initial_state=init_state,
                                                 sequence_length=seq_len,
                                                 scope=scope)
    return rnn_outputs, final_state


def get_reg(alphas, lam=0, type=""):
    alphas_loss = alphas + 1e-10
    reg = 0
    if type == "entropy":
        print("using entropy regularization for attention weights")
        # Entropy regularization
        reg = lam * tf.reduce_mean(-tf.reduce_sum(alphas_loss * tf.log(alphas_loss), axis=1))

    elif type == "weight":
        print("using weight regularization for attention weights")
        # Weight regularization
        reg = lam * tf.reduce_mean(tf.reduce_sum(alphas_loss * alphas_loss, axis=1))

    else:
        print("using no regularization for attention weights")

    return reg


def get_model(args, init, vocab_size):

    model = init(args, vocab_size)
    return model


def get_additive_model(init, pred_model, keyword_model):
    return init(pred_model, keyword_model)
