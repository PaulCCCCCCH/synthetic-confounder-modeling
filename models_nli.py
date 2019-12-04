import tensorflow as tf
import numpy as np
import blocks
from model_utils import dense_layer, attention_layer, get_reg, lstm_layer, dropout
from models import Model


class AdditiveModel(object):
    def __init__(self, pred_model, keyword_model):
        self.pred_model = pred_model
        self.keyword_model = keyword_model
        assert self.pred_model.batch_size == self.keyword_model.batch_size
        self.batch_size = self.pred_model.batch_size
        self.use_alphas = False
        if self.pred_model.use_alphas:
            self.use_alphas = True
            self.alphas_hypo = self.pred_model.alphas_hypo
            self.alphas_prem = self.pred_model.alphas_prem
        elif self.keyword_model.use_alphas:
            self.use_alphas = True
            self.alphas_hypo = self.keyword_model.alphas_hypo
            self.alphas_prem = self.keyword_model.alphas_prem
        self.build_model()

    def build_model(self):
        self.logits = self.pred_model.logits + tf.stop_gradient(self.keyword_model.logits)
        self.y = tf.nn.softmax(self.logits)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.pred_model.y_holder, depth=3), logits=self.logits))

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred_model.y_holder, tf.argmax(self.pred_model.y, 1)), tf.float32))
        self.optimizer = tf.train.AdamOptimizer(self.pred_model.learning_rate, beta1=0.9, beta2=0.999)
        self.train_op = self.optimizer.minimize(self.cost)

    def train_for_step(self, sess, train_x, train_y, start_idx, step_size=50):
        step_loss = 0
        step_accuracy = 0
        for i in range(step_size):
            batch_idx = np.array(range(start_idx + i * self.batch_size, start_idx + (i + 1) * self.batch_size))
            batch_xs = train_x[batch_idx, :, :]
            batch_ys = train_y[batch_idx]
            batch_loss, _, batch_accuracy = sess.run([self.cost, self.train_op, self.accuracy],
                                                     feed_dict={self.pred_model.x_holder: batch_xs,
                                                                self.pred_model.y_holder: batch_ys,
                                                                self.keyword_model.x_holder: batch_xs,
                                                                self.keyword_model.y_holder: batch_ys})
            step_loss += batch_loss
            step_accuracy += batch_accuracy
        step_loss /= step_size
        step_accuracy /= step_size
        print("Current step loss: ", step_loss , "Current step accuracy: ", step_accuracy )
        return step_loss, step_accuracy

    def train_for_epoch(self, sess, train_x, train_y):
        # cur_state = sess.run(init_state)
        batches_per_epoch = train_x.shape[0] // self.batch_size
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for idx in range(batches_per_epoch):
            batch_idx = np.random.choice(train_x.shape[0], size=self.batch_size, replace=False)
            batch_xs = train_x[batch_idx, :]
            batch_ys = train_y[batch_idx]
            batch_loss, _, batch_accuracy = sess.run([self.cost, self.train_op, self.accuracy],
                                                     feed_dict={self.pred_model.x_holder: batch_xs,
                                                                self.pred_model.y_holder: batch_ys,
                                                                self.keyword_model.x_holder: batch_xs,
                                                                self.keyword_model.y_holder: batch_ys})
            if idx % 50 == 0:
                print("Current batch loss: ", batch_loss, "Current batch acc: ", batch_accuracy)
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
        return epoch_loss / batches_per_epoch, epoch_accuracy / batches_per_epoch

    def predict_no_alphas(self, sess, test_x):
        pred_y = sess.run(self.pred_model.y, feed_dict={self.pred_model.x_holder: test_x})
        return pred_y

    def predict(self, sess, test_x):
        pred_y, alphas_hypo, alphas_prem = sess.run([self.pred_model.y, self.pred_model.alphas_hypo, self.pred_model.alphas_prem], feed_dict={self.pred_model.x_holder: test_x})
        return pred_y, alphas_hypo, alphas_prem

    def evaluate_accuracy(self, sess, test_x, test_y):
        test_accuracy = 0.0
        test_batches = test_x.shape[0] // self.batch_size
        for i in range(test_batches):
            test_idx = range(i * self.batch_size, (i + 1) * self.batch_size)
            test_xs = test_x[test_idx, :]
            test_ys = test_y[test_idx]
            pred_ys = self.predict_no_alphas(sess, test_xs)
            test_accuracy += np.sum(np.argmax(pred_ys, axis=1) == test_ys)
        test_accuracy /= (test_batches * self.batch_size)
        return test_accuracy

    def evaluate_capturing(self, sess, test_x, test_y, effect_dict):
        raise NotImplementedError


class NLIModel(Model):

    def train_for_step(self, sess, train_x, train_y, start_idx, step_size=50):
        step_loss = 0
        step_accuracy = 0
        for i in range(step_size):
            # batch_idx = np.random.choice(train_x.shape[0], size=self.batch_size, replace=False)
            batch_idx = np.array(range(start_idx + i * self.batch_size, start_idx + (i + 1) * self.batch_size))
            batch_xs = train_x[batch_idx, :, :]
            batch_ys = train_y[batch_idx]
            batch_loss, _, batch_accuracy = sess.run([self.cost, self.train_op, self.accuracy],
                                                     feed_dict={self.x_holder: batch_xs,
                                                                self.y_holder: batch_ys})
            step_loss += batch_loss
            step_accuracy += batch_accuracy

        step_loss /= step_size
        step_accuracy /= step_size
        print("Current step loss: ", step_loss, "Current step accuracy: ", step_accuracy)
        return step_loss, step_accuracy


    def train_for_epoch(self, sess, train_x, train_y):
        # cur_state = sess.run(init_state)
        batches_per_epoch = train_x.shape[0] // self.batch_size
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for idx in range(batches_per_epoch):
            batch_idx = np.random.choice(train_x.shape[0], size=self.batch_size, replace=False)
            batch_xs = train_x[batch_idx, :, :]
            batch_ys = train_y[batch_idx]
            batch_loss, _, batch_accuracy = sess.run([self.cost, self.train_op, self.accuracy],
                                                     feed_dict={self.x_holder: batch_xs,
                                                                self.y_holder: batch_ys})
            if idx % 50 == 0:
                print("Current batch loss: ", batch_loss, "Current batch acc: ", batch_accuracy)

            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
        return epoch_loss / batches_per_epoch, epoch_accuracy / batches_per_epoch

    def predict(self, sess, test_x):
        pred_y, alphas_hypo, alphas_prem = sess.run([self.y, self.alphas_hypo, self.alphas_prem], feed_dict={self.x_holder: test_x})
        return pred_y, alphas_hypo, alphas_prem

    def evaluate_accuracy(self, sess, test_x, test_y):
        test_accuracy = 0.0
        test_batches = test_x.shape[0] // self.batch_size
        for i in range(test_batches):
            test_idx = range(i * self.batch_size, (i + 1) * self.batch_size)
            test_xs = test_x[test_idx, :, :]
            test_ys = test_y[test_idx]
            pred_ys = self.predict_no_alphas(sess, test_xs)
            test_accuracy += np.sum(np.argmax(pred_ys, axis=1) == test_ys)
        test_accuracy /= (test_batches * self.batch_size)
        return test_accuracy

    def build_inputs(self):
        # input shape = (batch_size, sentence_length, emb_dim)
        self.x_holder = tf.placeholder(tf.int32, shape=[None, 2, self.max_len])
        self.y_holder = tf.placeholder(tf.int64, shape=[None])
        #self.prem_seq_lengths = tf.cast(tf.reduce_sum(tf.sign(self.x_holder[:, 0, :]), axis=1), tf.int32)
        #self.hyp_seq_lengths = tf.cast(tf.reduce_sum(tf.sign(self.x_holder[:, 1, :]), axis=1), tf.int32)
        self.prem_seq_lengths, self.mask_prem = blocks.length(self.x_holder[:, 0, :])
        self.hyp_seq_lengths, self.mask_hyp = blocks.length(self.x_holder[:, 1, :])

    def build_embedding(self, drop=True):
        self.embedding_w = tf.get_variable('embed_w', shape=[self.vocab_size, self.emb_dim],
                                               initializer=tf.random_uniform_initializer())

        self.e_prem = dropout(tf.nn.embedding_lookup(self.embedding_w, self.x_holder[:, 1, :]), self.keep_probs)
        self.e_hypo = dropout(tf.nn.embedding_lookup(self.embedding_w, self.x_holder[:, 0, :]), self.keep_probs)


class RegAttention(NLIModel):

    def build_model(self):
        # input shape = (batch_size, sentence_length, emb_dim)

        self.use_alphas = True

        rnn_outputs_prem, final_state_prem = lstm_layer(self.e_prem, self.lstm_size, self.batch_size, self.prem_seq_lengths, "prem")
        rnn_outputs_hypo, final_state_hypo = lstm_layer(self.e_hypo, self.lstm_size, self.batch_size, self.hyp_seq_lengths, "hypo")

        last_output_prem, alphas_prem = attention_layer(self.attention_size, rnn_outputs_prem, "encoder_prem", sparse=self.sparse, mask=self.mask_prem)
        last_output_hypo, alphas_hypo = attention_layer(self.attention_size, rnn_outputs_hypo, "encoder_hypo", sparse=self.sparse, mask=self.mask_hyp)

        self.alphas_prem = alphas_prem
        self.alphas_hypo = alphas_hypo
        self.logits = dense_layer(tf.concat([last_output_prem, last_output_hypo], axis=1), 3, activation=None, name="pred_out")
        self.y = tf.nn.softmax(self.logits)

        # WARNING: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency.
        # Do not call this op with the output of softmax, as it will produce incorrect results.
        reg1 = get_reg(alphas_hypo, lam=self.lam, type=self.reg)
        reg2 = get_reg(alphas_prem, lam=self.lam, type=self.reg)
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_holder, logits=self.logits)) + reg1 + reg2

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
        self.train_op = self.optimizer.minimize(self.cost)


class LSTMPredModelWithMLPKeyWordModelAdvTrain(NLIModel):

    def build_model(self):
        self.use_alphas = True
        rnn_outputs_prem, final_state_prem = lstm_layer(self.e_prem, self.lstm_size, self.batch_size, self.prem_seq_lengths, "prem")
        rnn_outputs_hypo, final_state_hypo = lstm_layer(self.e_hypo, self.lstm_size, self.batch_size, self.hyp_seq_lengths, "hypo")

        last_output_prem, alphas_prem = attention_layer(self.attention_size, rnn_outputs_prem, "encoder_prem", sparse=self.sparse, mask=self.prem_seq_lengths)
        last_output_hypo, alphas_hypo = attention_layer(self.attention_size, rnn_outputs_hypo, "encoder_hypo", sparse=self.sparse, mask=self.hyp_seq_lengths)
        self.alphas_prem = alphas_prem
        self.alphas_hypo = alphas_hypo
        self.logits = dense_layer(tf.concat([last_output_prem, last_output_hypo], axis=1), 3, activation=None, name="pred_out")
        self.y = tf.nn.softmax(self.logits)


        adv_in_prem = tf.reshape(self.e_prem, [-1, self.e_prem.shape[1] * self.e_hypo.shape[2]])
        adv_in_hypo = tf.reshape(self.e_hypo, [-1, self.e_hypo.shape[1] * self.e_hypo.shape[2]])
        """
        ### Debug ###
        self.w_adv = tf.get_variable("w", shape=[adv_in.shape[-1], 2],
                                     initializer=tf.truncated_normal_initializer())
        self.b_adv = tf.get_variable("b", shape=[2], dtype=tf.float32)

        adv_logits = tf.matmul(adv_in, self.w_adv) + self.b_adv
        ############
        """
        adv_logits = dense_layer(tf.concat([adv_in_hypo, adv_in_prem], axis=1), 3, activation=None, name="adv_encoder")
        adv_cost = 1 / tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_holder, depth=3), logits=adv_logits))

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_holder, depth=3), logits=self.logits))
        self.cost = self.cost + 0.01 * adv_cost

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
        self.train_op = self.optimizer.minimize(self.cost)

class LSTMPredModel(NLIModel):

    def build_model(self):
        self.use_alphas = True
        rnn_outputs_prem, final_state_prem = lstm_layer(self.e_prem, self.lstm_size, self.batch_size, self.prem_seq_lengths, "prem")
        rnn_outputs_hypo, final_state_hypo = lstm_layer(self.e_hypo, self.lstm_size, self.batch_size, self.hyp_seq_lengths, "hypo")

        last_output_prem, alphas_prem = attention_layer(self.attention_size, rnn_outputs_prem, "encoder_prem", sparse=self.sparse, mask=self.prem_seq_lengths)
        last_output_hypo, alphas_hypo = attention_layer(self.attention_size, rnn_outputs_hypo, "encoder_hypo", sparse=self.sparse, mask=self.hyp_seq_lengths)
        self.alphas_prem = alphas_prem
        self.alphas_hypo = alphas_hypo
        self.logits = dense_layer(tf.concat([last_output_prem, last_output_hypo], axis=1), 3, activation=None, name="pred_out")
        self.y = tf.nn.softmax(self.logits)

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_holder, logits=self.logits))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
        self.train_op = self.optimizer.minimize(self.cost)


class MLPPredModel(NLIModel):

    def build_model(self):
        inputs_prem = tf.reshape(self.e_prem, [-1, self.e_prem.shape[1] * self.e_prem.shape[2]])
        inputs_hypo = tf.reshape(self.e_hypo, [-1, self.e_hypo.shape[1] * self.e_hypo.shape[2]])

        self.logits = dense_layer(tf.concat([inputs_prem, inputs_hypo], axis=1), 3, activation=None, name="pred_out")

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_holder, logits=self.logits))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
        self.train_op = self.optimizer.minimize(self.cost)

        self.y = tf.nn.softmax(self.logits)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))


class LSTMPredModelWithRegAttentionKeyWordModelHEX(NLIModel):

    def build_model(self):
        self.use_alphas = True
        # Define prediction rnn
        rnn_outputs_prem, final_state_prem = lstm_layer(self.e_prem, self.lstm_size, self.batch_size, self.prem_seq_lengths, "prem")
        rnn_outputs_hypo, final_state_hypo = lstm_layer(self.e_hypo, self.lstm_size, self.batch_size, self.hyp_seq_lengths, "hypo")

        last_output_prem, alphas_prem = attention_layer(self.attention_size, rnn_outputs_prem, "encoder_prem", sparse=self.sparse, mask=self.prem_seq_lengths)
        last_output_hypo, alphas_hypo = attention_layer(self.attention_size, rnn_outputs_hypo, "encoder_hypo", sparse=self.sparse, mask=self.hyp_seq_lengths)
        self.alphas_prem = alphas_prem
        self.alphas_hypo = alphas_hypo
        # last_output = tf.nn.dropout(last_output, self.keep_probs)

        # Define key-word model rnn
        kwm_rnn_outputs_prem, kwm_final_state_prem = lstm_layer(self.e_prem, self.lstm_size, self.batch_size, self.prem_seq_lengths, scope="kwm_prem")
        kwm_last_output_prem, kwm_alphas_prem = attention_layer(self.attention_size, kwm_rnn_outputs_prem, "kwm_encoder_prem", sparse=self.sparse, mask=self.prem_seq_lengths)
        kwm_rnn_outputs_hypo, kwm_final_state_hypo = lstm_layer(self.e_hypo, self.lstm_size, self.batch_size, self.hyp_seq_lengths, scope="kwm_hypo")
        kwm_last_output_hypo, kwm_alphas_hypo = attention_layer(self.attention_size, kwm_rnn_outputs_hypo, "kwm_encoder_hypo", sparse=self.sparse, mask=self.hyp_seq_lengths)

        last_output = tf.concat([last_output_prem, last_output_hypo], axis=1)
        kwm_last_output = tf.concat([kwm_last_output_prem, kwm_last_output_hypo], axis=1)

        ############################
        # Hex #########################

        h_fc1 = last_output
        h_fc2 = kwm_last_output

        # Hex layer definition

        # Compute prediction using [h_fc1, 0(pad)]
        pad = tf.zeros_like(h_fc2, tf.float32)
        # print(pad.shape) -> (?, 600)

        yconv_contact_pred = tf.nn.dropout(tf.concat([h_fc1, pad], 1), self.keep_probs)

        # y_conv_pred = tf.matmul(yconv_contact_pred, self.W_cl) + self.b_cl
        y_conv_pred = dense_layer(yconv_contact_pred, 3, name="conv_pred")

        self.logits = y_conv_pred  # Prediction

        # Compute loss using [h_fc1, h_fc2] and [0(pad2), h_fc2]
        pad2 = tf.zeros_like(h_fc1, tf.float32)

        yconv_contact_H = tf.concat([pad2, h_fc2], 1)
        # Get Fg
        # y_conv_H = tf.matmul(yconv_contact_H, self.W_cl) + self.b_cl  # get Fg
        y_conv_H = dense_layer(yconv_contact_H, 3, name="conv_H")

        yconv_contact_loss = tf.nn.dropout(tf.concat([h_fc1, h_fc2], 1), self.keep_probs)
        # Get Fb
        # y_conv_loss = tf.matmul(yconv_contact_loss, self.W_cl) + self.b_cl  # get Fb
        y_conv_loss = dense_layer(yconv_contact_loss, 3, name="conv_loss")

        temp = tf.matmul(y_conv_H, y_conv_H, transpose_a=True)
        self.temp = temp

        y_conv_loss = y_conv_loss - tf.matmul(
            tf.matmul(tf.matmul(y_conv_H, tf.matrix_inverse(temp)), y_conv_H, transpose_b=True),
            y_conv_loss)  # get loss

        self.logits = y_conv_loss
        self.y = tf.nn.softmax(self.logits)



        # Regularize kwm attention
        reg1 = get_reg(kwm_alphas_hypo, lam=self.lam, type=self.reg)
        reg2 = get_reg(kwm_alphas_prem, lam=self.lam, type=self.reg)

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_holder, logits=self.logits)) + reg1 + reg2

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
        self.train_op = self.optimizer.minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))


class BiLSTMPredModel(NLIModel):

    def build_model(self):

        ## Define parameters

        self.W_mlp = tf.Variable(tf.random_normal([self.emb_dim * 8, self.emb_dim], stddev=0.1))
        self.b_mlp = tf.Variable(tf.random_normal([self.emb_dim], stddev=0.1))

        self.W_cl = tf.Variable(tf.random_normal([self.emb_dim, 3], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))

        # Get lengths of unpadded sentences
        prem_seq_lengths, prem_mask = blocks.length(self.x_holder[:, 0, :])
        hyp_seq_lengths, hyp_mask = blocks.length(self.x_holder[:, 1, :])

        ### BiLSTM layer ###
        premise_in = tf.nn.dropout(self.e_prem, keep_prob=self.keep_probs)
        hypothesis_in = tf.nn.dropout(self.e_hypo, keep_prob=self.keep_probs)

        premise_outs, c1 = blocks.biLSTM(premise_in, dim=self.emb_dim, seq_len=prem_seq_lengths, name='premise')
        hypothesis_outs, c2 = blocks.biLSTM(hypothesis_in, dim=self.emb_dim, seq_len=hyp_seq_lengths, name='hypothesis')

        premise_bi = tf.concat(premise_outs, axis=2)
        hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

        # premise_final = blocks.last_output(premise_bi, prem_seq_lengths)
        # hypothesis_final =  blocks.last_output(hypothesis_bi, hyp_seq_lengths)

        ### Mean pooling
        premise_sum = tf.reduce_sum(premise_bi, 1)
        premise_ave = tf.div(premise_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1))

        hypothesis_sum = tf.reduce_sum(hypothesis_bi, 1)
        hypothesis_ave = tf.div(hypothesis_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1))

        ### Mou et al. concat layer ###
        diff = tf.subtract(premise_ave, hypothesis_ave)
        mul = tf.multiply(premise_ave, hypothesis_ave)
        h = tf.concat([premise_ave, hypothesis_ave, diff, mul], 1)

        # MLP layer
        h_mlp = tf.nn.relu(tf.matmul(h, self.W_mlp) + self.b_mlp)
        # Dropout applied to classifier
        h_drop = tf.nn.dropout(h_mlp, self.keep_probs)

        # Get prediction
        self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

        # Define the cost function
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_holder, logits=self.logits))
        self.y = tf.nn.softmax(self.logits)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
        self.train_op = self.optimizer.minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))
        

class BiLSTMAttentionPredModel(NLIModel):

    def build_model(self):
        ## Define parameters

        self.W_mlp = tf.Variable(tf.random_normal([self.emb_dim * 4, self.emb_dim], stddev=0.1))
        self.b_mlp = tf.Variable(tf.random_normal([self.emb_dim], stddev=0.1))

        self.W_cl = tf.Variable(tf.random_normal([self.emb_dim, 3], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))

        # Get lengths of unpadded sentences
        prem_seq_lengths, prem_mask = blocks.length(self.x_holder[:, 0, :])
        hyp_seq_lengths, hyp_mask = blocks.length(self.x_holder[:, 1, :])

        ### BiLSTM layer ###
        premise_in = tf.nn.dropout(self.e_prem, keep_prob=self.keep_probs)
        hypothesis_in = tf.nn.dropout(self.e_hypo, keep_prob=self.keep_probs)

        premise_outs, premise_final = blocks.biLSTM(premise_in, dim=self.emb_dim, seq_len=prem_seq_lengths, name='premise')
        attention_outs_pre, self.alphas_pre = attention_layer(self.attention_size, premise_outs, 'prem_encoder_attention', sparse=self.sparse, mask=self.prem_seq_lengths)
        drop_pre = tf.nn.dropout(attention_outs_pre, self.keep_probs)
        # drop_pre = attention_outs_pre

        hypothesis_outs, hypothesis_final = blocks.biLSTM(hypothesis_in, dim=self.emb_dim, seq_len=hyp_seq_lengths,
                                                          name='hypothesis')
        attention_outs_hyp, self.alphas_hyp = attention_layer(self.attention_size, hypothesis_outs, 'hypo_encoder_attention', sparse=self.sparse, mask=self.hyp_seq_lengths)
        drop_hyp = tf.nn.dropout(attention_outs_hyp, self.keep_probs)
        # drop_hyp = attention_outs_hyp

        # Concat output of pre and hyp outpuratet
        drop = tf.concat([drop_pre, drop_hyp], axis=1)

        # MLP layer
        h_mlp = tf.nn.relu(tf.matmul(drop, self.W_mlp) + self.b_mlp)

        # Get prediction
        self.logits = tf.matmul(h_mlp, self.W_cl) + self.b_cl

        reg1 = get_reg(self.alphas_pre, lam=self.lam, type=self.reg)
        reg2 = get_reg(self.alphas_hyp, lam=self.lam, type=self.reg)

        # Define the cost function
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_holder, depth=3), logits=self.logits)) + reg1 + reg2
        self.y = tf.nn.softmax(self.logits)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
        self.train_op = self.optimizer.minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))


class ESIMPredModel(NLIModel):
    def build_model(self):

        ## Define parameters

        self.W_mlp = tf.Variable(tf.random_normal([self.emb_dim * 8, self.emb_dim], stddev=0.1))
        self.b_mlp = tf.Variable(tf.random_normal([self.emb_dim], stddev=0.1))

        self.W_cl = tf.Variable(tf.random_normal([self.emb_dim, 3], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))


        # Get lengths of unpadded sentences
        prem_seq_lengths, self.mask_prem = blocks.length(self.x_holder[:, 0, :])
        hyp_seq_lengths, self.mask_hyp = blocks.length(self.x_holder[:, 1, :])

        ### First biLSTM layer ###
        premise_in = tf.nn.dropout(self.e_prem, keep_prob=self.keep_probs)
        hypothesis_in = tf.nn.dropout(self.e_hypo, keep_prob=self.keep_probs)

        premise_outs, c1 = blocks.biLSTM(premise_in, dim=self.emb_dim, seq_len=prem_seq_lengths, name='premise')
        hypothesis_outs, c2 = blocks.biLSTM(hypothesis_in, dim=self.emb_dim, seq_len=hyp_seq_lengths, name='hypothesis')

        premise_bi = tf.concat(premise_outs, axis=2)
        hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

        premise_list = tf.unstack(premise_bi, axis=1)
        hypothesis_list = tf.unstack(hypothesis_bi, axis=1)

        ### Attention ###

        scores_all = []
        premise_attn = []
        alphas = []

        for i in range(self.max_len):

            scores_i_list = []
            for j in range(self.max_len):
                score_ij = tf.reduce_sum(tf.multiply(premise_list[i], hypothesis_list[j]), 1, keep_dims=True)
                scores_i_list.append(score_ij)

            scores_i = tf.stack(scores_i_list, axis=1)
            alpha_i = blocks.masked_softmax(scores_i, tf.expand_dims(self.mask_hyp, -1))
            a_tilde_i = tf.reduce_sum(tf.multiply(alpha_i, hypothesis_bi), 1)
            premise_attn.append(a_tilde_i)

            scores_all.append(scores_i)
            alphas.append(alpha_i)

        scores_stack = tf.stack(scores_all, axis=2)
        scores_list = tf.unstack(scores_stack, axis=1)

        hypothesis_attn = []
        betas = []
        for j in range(self.max_len):
            scores_j = scores_list[j]
            beta_j = blocks.masked_softmax(scores_j, tf.expand_dims(self.mask_prem, -1))
            b_tilde_j = tf.reduce_sum(tf.multiply(beta_j, premise_bi), 1)
            hypothesis_attn.append(b_tilde_j)

            betas.append(beta_j)

        # Make attention-weighted sentence representations into one tensor,
        premise_attns = tf.stack(premise_attn, axis=1)
        hypothesis_attns = tf.stack(hypothesis_attn, axis=1)

        # For making attention plots,
        self.alpha_s = tf.stack(alphas, axis=2)
        self.beta_s = tf.stack(betas, axis=2)

        ### Subcomponent Inference ###

        prem_diff = tf.subtract(premise_bi, premise_attns)
        prem_mul = tf.multiply(premise_bi, premise_attns)
        hyp_diff = tf.subtract(hypothesis_bi, hypothesis_attns)
        hyp_mul = tf.multiply(hypothesis_bi, hypothesis_attns)

        m_a = tf.concat([premise_bi, premise_attns, prem_diff, prem_mul], 2)
        m_b = tf.concat([hypothesis_bi, hypothesis_attns, hyp_diff, hyp_mul], 2)

        ### Inference Composition ###

        v1_outs, c3 = blocks.biLSTM(m_a, dim=self.emb_dim, seq_len=prem_seq_lengths, name='v1')
        v2_outs, c4 = blocks.biLSTM(m_b, dim=self.emb_dim, seq_len=hyp_seq_lengths, name='v2')

        v1_bi = tf.concat(v1_outs, axis=2)
        v2_bi = tf.concat(v2_outs, axis=2)

        ### Pooling Layer ###

        v_1_sum = tf.reduce_sum(v1_bi, 1)
        v_1_ave = tf.div(v_1_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1))

        v_2_sum = tf.reduce_sum(v2_bi, 1)
        v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1))

        v_1_max = tf.reduce_max(v1_bi, 1)
        v_2_max = tf.reduce_max(v2_bi, 1)

        v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)

        # MLP layer
        h_mlp = tf.nn.tanh(tf.matmul(v, self.W_mlp) + self.b_mlp)

        # Dropout applied to classifier
        h_drop = tf.nn.dropout(h_mlp, self.keep_probs)

        # Get prediction
        self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

        # Define the cost function

        # Define the cost function
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_holder, logits=self.logits))
        self.y = tf.nn.softmax(self.logits)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
        self.train_op = self.optimizer.minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))


class CBOWPredModel(NLIModel):
    def build_model(self):

        ## Define placeholders
        self.keep_rate_ph = tf.placeholder(tf.float32, [])

        self.W_0 = tf.Variable(tf.random_normal([self.emb_dim * 4, self.emb_dim], stddev=0.1), name="w0")
        self.b_0 = tf.Variable(tf.random_normal([self.emb_dim], stddev=0.1), name="b0")

        self.W_1 = tf.Variable(tf.random_normal([self.emb_dim, self.emb_dim], stddev=0.1), name="w1")
        self.b_1 = tf.Variable(tf.random_normal([self.emb_dim], stddev=0.1), name="b1")

        self.W_2 = tf.Variable(tf.random_normal([self.emb_dim, self.emb_dim], stddev=0.1), name="w2")
        self.b_2 = tf.Variable(tf.random_normal([self.emb_dim], stddev=0.1), name="b2")

        self.W_cl = tf.Variable(tf.random_normal([self.emb_dim, 3], stddev=0.1), name="wcl")
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1), name="bcl")


        premise_rep = tf.reduce_sum(self.e_prem, 1)
        hypothesis_rep = tf.reduce_sum(self.e_hypo, 1)

        ## Combinations
        h_diff = premise_rep - hypothesis_rep
        h_mul = premise_rep * hypothesis_rep

        ### MLP
        mlp_input = tf.concat([premise_rep, hypothesis_rep, h_diff, h_mul], 1)
        h_1 = tf.nn.relu(tf.matmul(mlp_input, self.W_0) + self.b_0)
        h_2 = tf.nn.relu(tf.matmul(h_1, self.W_1) + self.b_1)
        h_3 = tf.nn.relu(tf.matmul(h_2, self.W_2) + self.b_2)
        h_drop = tf.nn.dropout(h_3, self.keep_probs)

        # Get prediction
        self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl


        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_holder, logits=self.logits))
        self.y = tf.nn.softmax(self.logits)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
        self.train_op = self.optimizer.minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))
