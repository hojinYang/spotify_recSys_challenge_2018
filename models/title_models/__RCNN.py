# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class Model():
    def __init__(self, config, conv_layers):
        self.config = config
        self.embedding = config.emb
        self.input_len = config.strmaxlen
        self.output_dim = config.output
        self.y2_dim = config.y2_dim
        self.char_size = config.charsize
        self.rnn_hidden = config.rnn_hidden



    def fit(self, lr=0.01):
        with tf.name_scope("Input-Layer"):
            # Input
            self.x1 = tf.placeholder(tf.int64, shape=[None, self.input_len], name="input_x")
            self.y_ = tf.placeholder(tf.float32, shape=[None], name="output_x")
            keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            if self.embedding != 0:
                stdv = 1 / np.sqrt(self.embedding)
                embedding_matrix = tf.Variable(
                    tf.random_uniform(shape=[self.char_size, self.embedding], minval=-stdv, maxval=stdv), dtype=tf.float32,
                    name="Embedding_Weight")

        # EMBEDDING LAYERS
        with tf.name_scope("Embedding-Layer"):
            # CNN (tf.conv2d) : input = [batch, in_height, in_width, in_channels]
            # NLP embedding only has 1 channels
            # shape = (Batch, input_len, char_size, 1)
            if self.embedding == 0:
                # one hot encoding of zero vector, 0, 1, 2, ... , char_size - 1
                one_hot = tf.concat(
                    [
                        tf.zeros([1, self.char_size]),
                        tf.one_hot(list(range(self.char_size)), self.char_size, 1.0, 0.0)
                    ],
                    0
                )
                input = tf.nn.embedding_lookup(one_hot, self.x1)
            else:
                input = tf.nn.embedding_lookup(embedding_matrix, self.x1)
            input_shape = tf.shape(input)
            batch_size = input_shape[0]
            # Embedding Weight 후 (Batch, input_len, emb_dim)
            # zero_emb = tf.zeros([batch_size, 1, config.emb], dtype=tf.float32)
            zero_emb = tf.ones([batch_size, 1, self.rnn_hidden], dtype=tf.float32)

        with tf.name_scope("RNN-Layer"), tf.variable_scope("RNN-Input"):
            input_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_hidden)
            input_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_hidden)
            input_init_state = input_cell_1.zero_state(batch_size=input_shape[0], dtype=tf.float32)
            input_init_state = input_cell_2.zero_state(batch_size=input_shape[0], dtype=tf.float32)
            (left, right), _ = tf.nn.bidirectional_dynamic_rnn(input_cell_1, input_cell_2, input, dtype=tf.float32,
                                                               time_major=False)

        left_ctx = tf.concat((zero_emb, left[:, 1:, :]), axis=1)
        right_ctx = tf.concat((right[:, :-1, :], zero_emb), axis=1)
        recurrent_input = tf.concat((left_ctx, input, right_ctx), axis=2)

        with tf.name_scope("Recurrent-Layer"):
            # weight_h = self.rnn_hidden * 2 + self.embedding
            # W = tf.Variable(tf.random_uniform([weight_h, self.y2_dim], minval=-0.1, maxval=0.1), dtype=tf.float32,
            #                 name="y2_W")
            # b = tf.Variable(tf.random_uniform([self.y2_dim], minval=-0.1, maxval=0.1), dtype=tf.float32, name="y2_b")
            bias_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
            recurrent_out = tf.contrib.layers.fully_connected(recurrent_input, self.y2_dim, activation_fn=tf.nn.tanh,
                                                              biases_initializer=bias_init)

        # with tf.name_scope("RNN-Left"), tf.variable_scope("RNN-Left"):
        #     left_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_hidden)
        #     left_init_state = left_cell.zero_state(batch_size=input_shape[0], dtype=tf.float32)
        #     left_rnn_output = tf.nn.dynamic_rnn(left_cell, left_emb, dtype=tf.float32, time_major=False)
        #
        # with tf.name_scope("RNN-Right"), tf.variable_scope("RNN-Right"):
        #     right_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_hidden)
        #     right_init_state = right_cell.zero_state(batch_size=input_shape[0], dtype=tf.float32)
        #     right_rnn_output = tf.nn.dynamic_rnn(right_cell, right_emb_rev, dtype=tf.float32, time_major=False)

        with tf.name_scope("Max-Pooling-Layer"):
            recurrent_out = tf.expand_dims(recurrent_out, axis=-1)
            rec_shape = recurrent_out.shape
            # [Batch, seq, y2_dim, 1]
            ksize = [1, rec_shape[1], 1, 1]
            max_pool = tf.nn.max_pool(recurrent_out, ksize=ksize, strides=[1, 1, 1, 1],
                                      padding="VALID")  # ?, 1, y2_dim, 1
            pooled = tf.squeeze(max_pool, [1, 3])

        with tf.name_scope("Output-Layer"):
            # stdv = 1 / tf.sqrt(tf.float32(dims[-1]))
            stdv = 1 / 5
            weight_h = self.y2_dim
            W = tf.Variable(tf.random_uniform([weight_h, self.output_dim], minval=-stdv, maxval=stdv), dtype=tf.float32,
                            name="Output_W")
            b = tf.Variable(tf.random_uniform([self.output_dim], minval=-stdv, maxval=stdv), dtype=tf.float32,
                            name="Output_b")

            _output = tf.nn.xw_plus_b(pooled, W, b, name="output_rating")
            self.output_sigmoid = tf.squeeze(tf.sigmoid(_output) * 9 + 1)
            # output_rating = _output

        with tf.name_scope("Loss"):
            self.loss = tf.losses.mean_squared_error(predictions=self.output_sigmoid, labels=self.y_)

        # with tf.name_scope("Accuracy"):
        #     correct_predictions = tf.equal(self.prediction, self.input_y)  # Regression
        #     accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        global_step = tf.Variable(0, trainable=False)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_step = optimizer.minimize(self.loss, global_step=global_step)