# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class Char_LSTM():
    def __init__(self, config, fc_layers):
        self.config = config
        self.embedding = config.emb
        self.input_len = config.strmaxlen
        self.output_dim = config.n_output
        self.char_size = config.charsize
        self.fc_layers = fc_layers
        self.rnn_hidden = config.rnn_hidden
        self.bi = config.bi

        with tf.name_scope("Input-Layer"):
            # Input
            self.titles = tf.placeholder(tf.int64, shape=[None, self.input_len], name="input_x")
            self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            if self.embedding != 0:
                init = tf.contrib.layers.xavier_initializer(uniform=False)
                embedding_matrix = tf.get_variable('embedding_matrix', [self.char_size, self.embedding], initializer=init)

        # EMBEDDING LAYERS
        with tf.name_scope("Embedding-Layer"):

            # CNN (tf.conv2d) : input = [batch, in_height, in_width, in_channels]
            # NLP embedding only has 1 channels
            # shape = (Batch, input_len, char_size, 1)
            if self.embedding == 0:
                rnn_x = tf.one_hot(self.titles, depth = self.char_size)
            else:
                rnn_x = tf.nn.embedding_lookup(embedding_matrix, self.titles)
            # Embedding Weight 후 (Batch, input_len, emb_dim)

        input_shape = tf.shape(rnn_x)

        # LSTM Layer
        with tf.name_scope('RNN-Layer'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_hidden)
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
            init_state_fw = cell.zero_state(batch_size=input_shape[0], dtype=tf.float32)
            if self.bi:
                cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_hidden)
                # cell2 = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8)
                init_state_bw = cell.zero_state(batch_size=input_shape[0], dtype=tf.float32)

                output, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell2, rnn_x, 
                                                            initial_state_fw=init_state_fw, 
                                                            initial_state_bw=init_state_bw, time_major=False)
                output = tf.concat(output, 2)
            else:
                output, _ = tf.nn.dynamic_rnn(cell, rnn_x, initial_state=init_state_fw, time_major=False)

            output_shape = output.get_shape()

            d = output_shape[-1].value
            # rnn_output = output[:, -1, :]                   # Use Last Hidden Only  (Batch x Hidden)
            rnn_output = tf.reduce_mean(output, axis=1)     # Use Mean Pooling      (Batch x Hidden)

            # d = output_shape[1].value * output_shape[2].value
            # rnn_output = tf.reshape(output, [-1, d])    # Use All Hidden      (Batch x d)

        fc_input = rnn_output

        for i, fc_dim in enumerate(self.fc_layers):
            with tf.name_scope("FC-Layer-" + str(i)):
                init = tf.contrib.layers.xavier_initializer(uniform=False)
                W = tf.get_variable("FC_W" + str(i), [d, fc_dim], initializer=init)
                b = tf.get_variable("FC_b" + str(i), [fc_dim], initializer=init)

                fc_input = tf.nn.xw_plus_b(fc_input, W, b, name="FC"+str(i))
                fc_input = tf.nn.dropout(tf.nn.relu(fc_input), self.keep_prob)
                d = fc_dim

        fc_output = fc_input

        with tf.name_scope("Output-Layer"):
            init = tf.contrib.layers.xavier_initializer(uniform=False)
            W = tf.get_variable("Output_W", [d, self.output_dim], initializer=init)
            b = tf.get_variable("Output_b", [self.output_dim], initializer=init)

            self.output = tf.sigmoid(tf.nn.xw_plus_b(fc_output, W, b, name="output_prob"))

    def __str__(self):
        name = "LSTM"
        bi = "Bidirectional : " + str(self.bi)
        embeddings = "Embedding Size : " + str(self.embedding)
        hidden = "RNN HIDDEN : " + str(self.rnn_hidden)
        layers = "FC Layers : " + str(self.fc_layers)

        return '\n'.join([name, bi, embeddings, hidden, layers])