# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class Model():
    def __init__(self, config, conv_layers):
        self.config = config
        self.embedding = config.emb
        self.input_len = config.strmaxlen
        self.output_dim = config.output
        self.char_size = config.charsize
        self.bi_directional = config.bi
        self.rnn_hidden = config.rnn_hidden
        self.conv_layers = conv_layers
        self.lr = config.lr
        with tf.name_scope("Input-Layer"):
            # Input
            self.x1 = tf.placeholder(tf.int64, shape=[None, self.input_len], name="input_x")
            self.y_ = tf.placeholder(tf.float32, shape=[None], name="output_x")
            # self.loss_weight = tf.placeholder(tf.float32, shape=[None], name="loss_weight")
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
                cnn_x = tf.nn.embedding_lookup(one_hot, self.x1)
            else:
                cnn_x = tf.nn.embedding_lookup(embedding_matrix, self.x1)
            # Embedding Weight 후 (Batch, input_len, emb_dim)

            cnn_x = tf.expand_dims(cnn_x, -1)

        check_point = 0

        # CONVOLUTION LAYERS
        for i, conv_info in enumerate(self.conv_layers):
            # conv_info = [# of feature, kernel height, pool height]
            check_point += 1

            with tf.name_scope("Conv-Layer" + str(i)):
                filter_width = cnn_x.get_shape()[2].value
                filter_shape = [conv_info[1], filter_width, 1, conv_info[0]]

                stdv = 1 / np.sqrt(conv_info[0] * conv_info[1])
                W = tf.Variable(tf.random_uniform(shape=filter_shape, minval=-stdv, maxval=stdv), dtype=tf.float32,
                                name='Conv_W')
                b = tf.Variable(tf.random_uniform(shape=[conv_info[0]], minval=-stdv, maxval=stdv), dtype=tf.float32,
                                name='Conv_b')
                # batch x seq_length x emb x 1
                # Temporal convolution : Filter width = # of features
                # filter_shape = [conv_info[1], filter_width, 1, conv_info[0]]
                conv = tf.nn.conv2d(cnn_x, W, [1, 1, 1, 1], "VALID", name="conv")
                cnn_x = tf.nn.bias_add(conv, b)

            with tf.name_scope("Non-Linear"):
                cnn_x = tf.nn.relu(cnn_x)
            if conv_info[-1] != -1:
                with tf.name_scope("Max-Polling"):
                    pool_shape = [1, conv_info[-1], 1, 1]
                    pool = tf.nn.max_pool(cnn_x, ksize=pool_shape, strides=pool_shape, padding="VALID")
                    cnn_x = tf.transpose(pool, [0, 1, 3, 2])
            else:
                cnn_x = tf.transpose(cnn_x, [0, 1, 3, 2])
        cnn_output = tf.squeeze(cnn_x, axis=3)

        with tf.name_scope("RNN-Layer"):
            cnn_shape = tf.shape(cnn_output)

            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_hidden)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.3)
            init_state_fw = cell.zero_state(batch_size=cnn_shape[0], dtype=tf.float32)
            if self.bi_directional:
                cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_hidden)
                cell_2 = tf.nn.rnn_cell.DropoutWrapper(cell_2, output_keep_prob=0.3)
                init_state_bw = cell_2.zero_state(batch_size=cnn_shape[0], dtype=tf.float32)
                output, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_2, cnn_output, dtype=tf.float32,
                                                            time_major=False)
                output = tf.concat(output, 2)
            else:
                output, _ = tf.nn.dynamic_rnn(cell, cnn_output, dtype=tf.float32, time_major=True)
            rnn_output = output[:, -1, :]

        dims = tf.shape(rnn_output)

        with tf.name_scope("Output-Layer"):
            # stdv = 1 / tf.sqrt(tf.float32(dims[-1]))
            stdv = 1 / 5
            weight_h = self.rnn_hidden * 2 if self.bi_directional else self.rnn_hidden
            W = tf.Variable(tf.random_uniform([weight_h, self.output_dim], minval=-stdv, maxval=stdv), dtype=tf.float32,
                            name="Output_W")
            b = tf.Variable(tf.random_uniform([self.output_dim], minval=-stdv, maxval=stdv), dtype=tf.float32,
                            name="Output_b")

            _output = tf.nn.xw_plus_b(rnn_output, W, b, name="output_prob")
            self.output_sigmoid = tf.squeeze(tf.sigmoid(_output) * 9 + 1)
            # output_rating = _output
        # flat_dim = cnn_output.shape[1].value * cnn_output.shape[2].value
        # output_in = tf.reshape(cnn_output, [-1, flat_dim])
        # with tf.name_scope("Output-Layer"):
        #     # stdv = 1 / tf.sqrt(tf.float32(dims[-1]))
        #     stdv = 1 / 5
        #     W = tf.Variable(tf.random_uniform([flat_dim, self.output_dim], minval=-stdv, maxval=stdv), dtype=tf.float32,
        #                     name="Output_W")
        #     b = tf.Variable(tf.random_uniform([self.output_dim], minval=-stdv, maxval=stdv), dtype=tf.float32,
        #                     name="Output_b")
        #
        #     _output = tf.nn.xw_plus_b(output_in, W, b, name="output_prob")
        #     self.output_sigmoid = tf.squeeze(tf.sigmoid(_output) * 9 + 1)
        #     # output_rating = _output

        with tf.name_scope("Loss"):
            # sq_err = tf.square(tf.subtract(self.output_sigmoid, self.y_))
            # self.loss = tf.reduce_mean(sq_err)
            self.loss = tf.losses.mean_squared_error(labels=self.y_, predictions=self.output_sigmoid)

        # with tf.name_scope("Accuracy"):
        #     correct_predictions = tf.equal(self.prediction, self.input_y)  # Regression
        #     accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        global_step = tf.Variable(0, trainable=False)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = optimizer.minimize(self.loss, global_step=global_step)

    def fit(self):
        pass
