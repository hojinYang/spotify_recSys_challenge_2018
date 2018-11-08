# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class Char_CNN():
    def __init__(self, config, conv_layers):
        self.config = config
        self.embedding = config.char_emb
        self.input_len = config.strmaxlen
        self.output_dim = config.n_output
        self.char_size = config.charsize
        self.conv_layers = conv_layers

        with tf.name_scope("Input-Layer"):
            # Input
            self.titles = tf.placeholder(tf.int64, shape=[None, self.input_len], name="input_x")
            self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            if self.embedding != 0:
                init = tf.contrib.layers.xavier_initializer(uniform=False)
                embedding_matrix = tf.get_variable('char_embedding', [self.char_size, self.embedding], initializer=init)

        # EMBEDDING LAYERS
        with tf.name_scope("Embedding-Layer"):
            if self.embedding == 0: #CharCNN의 방식
                # one hot encoding of zero vector, 0, 1, 2, ... , char_size - 1
                cnn_x = tf.one_hot(self.titles, depth=self.char_size)
            else: #VDCNN의 방식
                cnn_x = tf.nn.embedding_lookup(embedding_matrix, self.titles)
            # Embedding Weight 후 (Batch, input_len, emb_dim)

            cnn_x = tf.expand_dims(cnn_x, -1)

        check_point = 0

        cnn_output = None
        # WIDE CONVOLUTION LAYERS
        for i, conv_info in enumerate(self.conv_layers):
            # conv_info = [# of feature, kernel height, pool height]
            check_point += 1

            with tf.name_scope("Conv-Layer-" + str(i)):
                filter_width = cnn_x.get_shape()[2].value
                filter_shape = [conv_info[1], filter_width, 1, conv_info[0]]  # [각 filter 크기, emb 크기, 1, kernel 크기]

                init = tf.contrib.layers.xavier_initializer(uniform=False)
                W = tf.get_variable("Conv_W" + str(i), filter_shape, initializer=init)
                b = tf.get_variable("Conv_b" + str(i), [conv_info[0]], initializer=init)

                conv = tf.nn.conv2d(cnn_x, W, [1, 1, 1, 1], "VALID", name="conv")

            with tf.name_scope("Non-Linear"):
                conv = tf.nn.bias_add(conv, b)
                conv = tf.nn.relu(conv)
            if conv_info[-1] != -1:
                with tf.name_scope("Max-Polling"):
                    pool_shape = [1, conv_info[-1], 1, 1]
                    conv = tf.nn.max_pool(conv, ksize=pool_shape, strides=pool_shape, padding="VALID")
            with tf.name_scope("One-Max-Pooling"):
                conv = tf.reduce_max(conv, reduction_indices=[1], keep_dims=True)  # 1-max pooling
                conv = tf.squeeze(conv, [1, 2])
                if i == 0:
                    cnn_output = conv
                else:
                    cnn_output = tf.concat([cnn_output, conv], 1)


        d = cnn_output.shape[1].value
        cnn_output = tf.nn.dropout(cnn_output, self.keep_prob)
        with tf.name_scope("Output-Layer"):

            init = tf.contrib.layers.xavier_initializer(uniform=False)
            W = tf.get_variable("Output_W", [d, self.output_dim], initializer=init)
            b = tf.get_variable("Output_b", [self.output_dim], initializer=init)

            self.output = tf.sigmoid(tf.nn.xw_plus_b(cnn_output, W, b, name="output_prob"))

    def __str__(self):
        name = "Wide CNN"
        embeddings = "Embedding Size : " + str(self.embedding)
        filter_num = "Number of Filters : " + str(self.config.filter_num)
        layers = "Conv Layers : " + str(self.conv_layers)

        return '\n'.join([name, embeddings, filter_num, layers])