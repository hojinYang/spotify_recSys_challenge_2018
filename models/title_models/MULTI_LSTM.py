# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class MULTI_LSTM():
    def __init__(self, config, fc_layers, rnn_layers):
        self.config = config
        self.embedding = config.emb
        self.input_len = config.strmaxlen
        self.output_dim = config.output
        self.char_size = config.charsize
        self.fc_layers = fc_layers
        self.rnn_hidden = config.rnn_hidden
        self.rnn_layers = rnn_layers
        self.bi = config.bi


        with tf.name_scope("Input-Layer"):
            # Input
            self.x1 = tf.placeholder(tf.int64, shape=[None, self.input_len], name="input_x")
            self.y_ = tf.placeholder(tf.int64, shape=[None], name="output_x")
            keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            if self.embedding != 0:
                # stdv = 1 / np.sqrt(self.embedding)
                init = tf.contrib.layers.xavier_initializer(uniform=False)
                embedding_matrix = tf.get_variable('embedding_matrix', [self.char_size, self.embedding], initializer=init)


        # EMBEDDING LAYERS
        with tf.name_scope("Embedding-Layer"):
            # one hot encoding of zero vector, 0, 1, 2, ... , char_size - 1
            one_hot = tf.concat(
                [
                    tf.zeros([1, self.char_size]),
                    tf.one_hot(list(range(self.char_size)), self.char_size, 1.0, 0.0)
                ],
                0
            )
            # CNN (tf.conv2d) : input = [batch, in_height, in_width, in_channels]
            # NLP embedding only has 1 channels
            # shape = (Batch, input_len, char_size, 1)
            if self.embedding == 0:
                rnn_x = tf.nn.embedding_lookup(one_hot, self.x1)
            else:
                rnn_x = tf.nn.embedding_lookup(embedding_matrix, self.x1)
            # Embedding Weight 후 (Batch, input_len, emb_dim)

        input_shape = tf.shape(rnn_x)

        # LSTM Layer
        with tf.name_scope('RNN-Layer'):
            num_layers = len(self.rnn_layers)
            cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n) for n in self.rnn_layers]
            # cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_hidden)
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
            init_state_fw = multi_cell.zero_state(batch_size=input_shape[0], dtype=tf.float32)
            if self.bi:
                cells2 = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n) for n in self.rnn_layers]
                # cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_hidden)
                multi_cell2 = tf.nn.rnn_cell.MultiRNNCell(cells)
                # cell2 = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
                init_state_bw = multi_cell.zero_state(batch_size=input_shape[0], dtype=tf.float32)

                output, _ = tf.nn.bidirectional_dynamic_rnn(multi_cell, multi_cell2, rnn_x, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw, time_major=False)
                output = tf.concat(output, 2)
            else:
                output, _ = tf.nn.dynamic_rnn(multi_cell, rnn_x, initial_state=init_state_fw, time_major=False)

            output_shape = output.get_shape()

            d = output_shape[-1].value
            # rnn_output = output[:, -1, :]                   # Use Last Hidden Only  (Batch x Hidden)
            rnn_output = tf.reduce_mean(output, axis=1)     # Use Mean Pooling      (Batch x Hidden)

            # d = output_shape[1].value * output_shape[2].value
            # rnn_output = tf.reshape(output, [-1, d])    # Use All Hidden      (Batch x d)


        fc_input = rnn_output
        # [2048, 2048]
        for i, fc_dim in enumerate(self.fc_layers):
            with tf.name_scope("FC-Layer-" + str(i)):
                init = tf.contrib.layers.xavier_initializer(uniform=False)
                W = tf.get_variable("FC_W" + str(i), [d, fc_dim], initializer=init)
                b = tf.get_variable("FC_b" + str(i), [fc_dim], initializer=init)

                fc_input = tf.nn.xw_plus_b(fc_input, W, b, name="FC"+str(i))
                fc_input = tf.nn.dropout(tf.nn.relu(fc_input), 0.5)
                d = fc_dim

        fc_output = fc_input

        with tf.name_scope("Output-Layer"):
            init = tf.contrib.layers.xavier_initializer(uniform=False)
            W = tf.get_variable("Output_W", [d, self.output_dim], initializer=init)
            b = tf.get_variable("Output_b", [self.output_dim], initializer=init)

            self.output = tf.sigmoid(tf.nn.xw_plus_b(fc_output, W, b, name="output_prob"))

    def __str__(self):
        name = "Multilayer LSTM"
        bi = "Bidirectional : " + str(self.bi)
        embeddings = "Embedding Size : " + str(self.embedding)
        hidden = "RNN HIDDEN : " + str(self.rnn_hidden)
        num_layers = "Number of Layers : " + str(len(self.rnn_layers))
        layers = "FC Layers : " + str(self.fc_layers)

        return '\n'.join([name, bi, embeddings, hidden, num_layers, layers])