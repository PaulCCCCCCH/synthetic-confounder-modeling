import tensorflow as tf
import numpy as np


# Input must be of shape (Batch, TimeStep, HiddenSize)
def attention_layer(inputs, attention_size):
    hidden_size = inputs[2]


class Model(object):
    def __init__(self, batch_size=10, max_len=30, vocab_size=10000, embeddings_dim=20, keep_probs=0.9,
                 is_train=True, use_embedding=True):
        self.batch_size = batch_size
        self.max_len = max_len
        self.keep_probs = keep_probs
        self.emb_dim = embeddings_dim
        self.is_train = is_train
        self.vocab_size = vocab_size
        self.use_embedding = use_embedding

    def build_model(self):
        raise NotImplementedError

    def train_for_epoch(self, sess, train_x, train_y):
        # cur_state = sess.run(init_state)
        assert self.is_train, 'Not training model'
        batches_per_epoch = train_x.shape[0] // self.batch_size
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for idx in range(batches_per_epoch):
            batch_idx = np.random.choice(train_x.shape[0], size=self.batch_size, replace=False)
            batch_xs = train_x[batch_idx, :]
            batch_ys = train_y[batch_idx]
            batch_loss, _, batch_accuracy, temp = sess.run([self.cost, self.train_op, self.accuracy, self.alphas],
                                                           feed_dict={self.x_holder: batch_xs,
                                                                      self.y_holder: batch_ys})
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
        return epoch_loss / batches_per_epoch, epoch_accuracy / batches_per_epoch

    def predict(self, sess, test_x):
        pred_y, alphas = sess.run([self.y, self.alphas], feed_dict={self.x_holder: test_x})
        return pred_y, alphas

    def evaluate_accuracy(self, sess, test_x, test_y):
        test_accuracy = 0.0
        test_batches = test_x.shape[0] // self.batch_size
        for i in range(test_batches):
            test_idx = range(i * self.batch_size, (i + 1) * self.batch_size)
            test_xs = test_x[test_idx, :]
            test_ys = test_y[test_idx]
            pred_ys, alphas = self.predict(sess, test_xs)
            test_accuracy += np.sum(np.argmax(pred_ys, axis=1) == test_ys)
        test_accuracy /= (test_batches * self.batch_size)
        return test_accuracy

    def evaluate_capturing(self, sess, test_x, test_y, effect_dict):
        test_batches = test_x.shape[0] // self.batch_size
        score = 0
        for i in range(test_batches):
            test_idx = range(i * self.batch_size, (i + 1) * self.batch_size)
            test_xs = test_x[test_idx, :]
            test_ys = test_y[test_idx]
            pred_ys, alphas = self.predict(sess, test_xs)
            effects = np.asarray([[effect_dict[w] for w in sentence] for sentence in test_xs])
            factor = np.mean(np.sum(np.multiply(alphas, effects), axis=1))
            score += factor
        return score / test_batches

class SentimentModelWithRegAttention(Model):
    def __init__(self, batch_size=10, max_len=30, lstm_size=20, vocab_size=10000, embeddings_dim=20, keep_probs=0.9, is_train=True, attention_size=16, use_embedding=True, reg="none", lam=None):
        Model.__init__(self, batch_size, max_len, vocab_size, embeddings_dim, keep_probs, is_train, use_embedding)

        self.lstm_size = lstm_size
        self.attention_size = attention_size
        self.reg = reg
        self.lam = lam
        self.epsilon = 1e-10
        self.build_model()

    def build_model(self):
        # shape = (batch_size, sentence_length, emb_dim)
        self.x_holder = tf.placeholder(tf.int32, shape=[None, self.max_len])
        self.y_holder = tf.placeholder(tf.int64, shape=[None])
        self.seq_len = tf.cast(tf.reduce_sum(tf.sign(self.x_holder), axis=1), tf.int32)

        if self.use_embedding:
            self.embedding_w = tf.get_variable('embed_w', shape=[self.vocab_size,self.emb_dim], initializer=tf.random_uniform_initializer(), trainable=False)
            self.e = tf.nn.embedding_lookup(self.embedding_w, self.x_holder)
        else:
            self.embedding_w = tf.one_hot(list(range(self.vocab_size)), depth=self.vocab_size)
            self.e = tf.nn.embedding_lookup(self.embedding_w, self.x_holder)

        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        """
        if self.is_train:
            # lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, keep_probs=0.5)
            self.x_holder = tf.nn.dropout(self.x_holder, self.keep_probs)
        """
        self.init_state = lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=lstm,
                                                    inputs=self.e,
                                                    initial_state=self.init_state,
                                                    sequence_length=self.seq_len)

        self.w_omega = tf.get_variable('w_omega', initializer=tf.random_normal([self.lstm_size, self.attention_size]))
        self.b_omega = tf.get_variable('b_omega', initializer=tf.random_normal([self.attention_size]))
        self.u_omega = tf.get_variable('u_omega', initializer=tf.random_normal([self.attention_size]))

        self.value = tf.tanh(tf.tensordot(rnn_outputs, self.w_omega, axes=1) + self.b_omega)

        self.vu = tf.tensordot(self.value, self.u_omega, axes=1, name='vu')
        self.alphas = tf.nn.softmax(self.vu, name='alphas')

        last_output = tf.reduce_sum(rnn_outputs * tf.expand_dims(self.alphas, -1), 1)
        """
        batch_size = tf.shape(rnn_outputs)[0]
        max_length = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (self.seq_len - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        relevant = tf.gather(flat, index)
        #last_output = rnn_outputs[:,-1,:]
        relevant = tf.reduce_mean(rnn_outputs, axis=1)
        #last_output = tf.nn.dropout(last_output, 0.25)
        last_output = relevant
        """

        if self.is_train:
            last_output = tf.nn.dropout(last_output, self.keep_probs)
        self.w = tf.get_variable("w", shape=[self.lstm_size, 2], initializer=tf.truncated_normal_initializer())
        self.b = tf.get_variable("b", shape=[2], dtype=tf.float32)
        self.logits = tf.matmul(last_output, self.w) + self.b
        self.y = tf.nn.softmax(self.logits)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_holder, depth=2),logits=self.y))

        if self.reg == "entropy":
            print("using entropy regularization")
            # Entropy regularization
            alphas_loss = self.alphas + self.epsilon
            reg = self.lam * tf.reduce_mean(-tf.reduce_sum(alphas_loss * tf.log(alphas_loss), axis=1))
            self.cost = self.cost + reg

        elif self.reg == "weight":
            print("using weight regularization")
            # Weight regularization
            alphas_loss = self.alphas + self.epsilon
            reg = self.lam * tf.reduce_mean(tf.reduce_sum(alphas_loss * alphas_loss), axis=1)
            self.cost = self.cost + reg

        else:
            print("using no regularization")

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))
        
        if self.is_train:
            #print(self.cost)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
            self.train_op = self.optimizer.minimize(self.cost)


class SentimentModelWithSparseAttention(Model):
    def __init__(self, batch_size=10, max_len=30, lstm_size=20, vocab_size=10000, embeddings_dim=20, keep_probs=0.9,
                 is_train=True, attention_size=16, use_embedding=True, reg="none", lam=None):
        Model.__init__(self, batch_size, max_len, vocab_size, embeddings_dim, keep_probs, is_train, use_embedding)

        self.lstm_size = lstm_size
        self.attention_size = attention_size
        self.reg = reg
        self.lam = lam
        self.build_model()

    def build_model(self):
        # shape = (batch_size, sentence_length, emb_dim)
        self.x_holder = tf.placeholder(tf.int32, shape=[None, self.max_len])
        self.y_holder = tf.placeholder(tf.int64, shape=[None])
        self.seq_len = tf.cast(tf.reduce_sum(tf.sign(self.x_holder), axis=1), tf.int32)

        if self.use_embedding:
            self.embedding_w = tf.get_variable('embed_w', shape=[self.vocab_size, self.emb_dim],
                                               initializer=tf.random_uniform_initializer(), trainable=False)
            self.e = tf.nn.embedding_lookup(self.embedding_w, self.x_holder)
        else:
            self.embedding_w = tf.one_hot(list(range(self.vocab_size)), depth=self.vocab_size)
            self.e = tf.nn.embedding_lookup(self.embedding_w, self.x_holder)

        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)

        self.init_state = lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=lstm,
                                                     inputs=self.e,
                                                     initial_state=self.init_state,
                                                     sequence_length=self.seq_len)

        self.w_omega = tf.get_variable('w_omega', initializer=tf.random_normal([self.lstm_size, self.attention_size]))
        self.b_omega = tf.get_variable('b_omega', initializer=tf.random_normal([self.attention_size]))
        self.u_omega = tf.get_variable('u_omega', initializer=tf.random_normal([self.attention_size]))

        self.value = tf.tanh(tf.tensordot(rnn_outputs, self.w_omega, axes=1) + self.b_omega)

        self.vu = tf.tensordot(self.value, self.u_omega, axes=1, name='vu')

        # Use sparsemax instead of softmax
        # self.alphas = tf.nn.softmax(self.vu, name='alphas')
        self.alphas = tf.contrib.sparsemax.sparsemax(self.vu, name='alphas')

        # Normalised attention
        #self.alphas = self.vu / tf.expand_dims(tf.reduce_sum(tf.math.multiply(self.vu, self.vu), axis=1), 1)

        last_output = tf.reduce_sum(rnn_outputs * tf.expand_dims(self.alphas, -1), 1)


        if self.is_train:
            last_output = tf.nn.dropout(last_output, self.keep_probs)
        self.w = tf.get_variable("w", shape=[self.lstm_size, 2], initializer=tf.truncated_normal_initializer())
        self.b = tf.get_variable("b", shape=[2], dtype=tf.float32)
        self.logits = tf.matmul(last_output, self.w) + self.b
        self.y = tf.nn.softmax(self.logits)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_holder, depth=2), logits=self.y))

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))

        if self.is_train:
            # print(self.cost)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
            self.train_op = self.optimizer.minimize(self.cost)

def _to_effect_list(batch, effect_list):
    return
