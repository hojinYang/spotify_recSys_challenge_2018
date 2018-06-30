# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class WD_CNN():
    def __init__(self, config, dconv_layers, wconv_layers, fc_layers):
        self.config = config
        self.embedding = config.emb
        self.input_len = config.strmaxlen
        self.output_dim = config.output
        self.char_size = config.charsize
        self.bi_directional = config.bi
        self.rnn_hidden = config.rnn_hidden
        self.conv_layers = dconv_layers
        self.wconv_layers = wconv_layers
        self.fc_layers = fc_layers
        self.lr = config.lr

        with tf.name_scope("Input-Layer"):
            # Input
            self.x1 = tf.placeholder(tf.int64, shape=[None, self.input_len], name="input_x")
            self.y_ = tf.placeholder(tf.int64, shape=[None], name="output_x")
            # self.loss_weight = tf.placeholder(tf.float32, shape=[None], name="loss_weight")
            keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            if self.embedding != 0:
                stdv = 1 / np.sqrt(self.embedding)
                init = tf.contrib.layers.xavier_initializer(uniform=False)
                embedding_matrix = tf.get_variable('char_embedding', [self.char_size, self.embedding], initializer=init)
                # embedding_matrix = tf.Variable(
                #     tf.random_uniform(shape=[self.char_size, self.embedding], minval=-stdv, maxval=stdv), dtype=tf.float32,
                #     name="Embedding_Weight")

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

        wconv = None
        # WIDE CONVOLUTION LAYERS
        for i, conv_info in enumerate(self.wconv_layers):
            # conv_info = [# of feature, kernel height, pool height]
            check_point += 1

            with tf.name_scope("Wide-Conv-Layer" + str(i)):
                filter_width = cnn_x.get_shape()[2].value
                filter_shape = [conv_info[1], filter_width, 1, conv_info[0]]

                init = tf.contrib.layers.xavier_initializer(uniform=False)
                W = tf.get_variable("WConv_W" + str(i), filter_shape, initializer=init)
                b = tf.get_variable("WConv_b" + str(i), [conv_info[0]], initializer=init)

                # stdv = 1 / np.sqrt(conv_info[0] * conv_info[1])
                # W = tf.Variable(tf.random_uniform(shape=filter_shape, minval=-stdv, maxval=stdv), dtype=tf.float32,
                #                 name='Conv_W')
                # b = tf.Variable(tf.random_uniform(shape=[conv_info[0]], minval=-stdv, maxval=stdv), dtype=tf.float32,
                #                 name='Conv_b')
                conv = tf.nn.conv2d(cnn_x, W, [1, 1, 1, 1], "VALID", name="wconv")
                conv = tf.nn.bias_add(conv, b)

            with tf.name_scope("Non-Linear"):
                _conv = tf.nn.relu(conv)

            if conv_info[-1] != -1:
                with tf.name_scope("Max-Polling"):
                    pool_shape = [1, conv_info[-1], 1, 1]
                    pooled = tf.nn.max_pool(_conv, ksize=pool_shape, strides=pool_shape, padding="VALID")
                    wconv = pooled if i == 0 else tf.concat((wconv, pooled), axis=1)
            else:
                wconv = _conv if i == 0 else tf.concat((wconv, _conv), axis=1)

        wconv = tf.transpose(wconv, [0, 1, 3, 2])

        check_point = 0
        # DEEP CONVOLUTION LAYERS
        dconv = cnn_x
        for i, conv_info in enumerate(self.conv_layers):
            # conv_info = [# of feature, kernel height, pool height]
            check_point += 1

            with tf.name_scope("Deep-Conv-Layer" + str(i)):
                filter_width = dconv.get_shape()[2].value
                filter_shape = [conv_info[1], filter_width, 1, conv_info[0]]

                init = tf.contrib.layers.xavier_initializer(uniform=False)
                W = tf.get_variable("DConv_W" + str(i), filter_shape, initializer=init)
                b = tf.get_variable("DConv_b" + str(i), [conv_info[0]], initializer=init)
                # stdv = 1 / np.sqrt(conv_info[0] * conv_info[1])
                # W = tf.Variable(tf.random_uniform(shape=filter_shape, minval=-stdv, maxval=stdv), dtype=tf.float32,
                #                 name='Conv_W')
                # b = tf.Variable(tf.random_uniform(shape=[conv_info[0]], minval=-stdv, maxval=stdv), dtype=tf.float32,
                #                 name='Conv_b')
                # batch x seq_length x emb x 1
                # Temporal convolution : Filter width = # of features
                # filter_shape = [conv_info[1], filter_width, 1, conv_info[0]]
                conv = tf.nn.conv2d(dconv, W, [1, 1, 1, 1], "VALID", name="dconv")
                dconv = tf.nn.bias_add(conv, b)

            with tf.name_scope("Non-Linear"):
                dconv = tf.nn.relu(dconv)
            if conv_info[-1] != -1:
                with tf.name_scope("Max-Polling"):
                    pool_shape = [1, conv_info[-1], 1, 1]
                    pool = tf.nn.max_pool(dconv, ksize=pool_shape, strides=pool_shape, padding="VALID")
                    dconv = tf.transpose(pool, [0, 1, 3, 2])
            else:
                dconv = tf.transpose(dconv, [0, 1, 3, 2])
        cnn_output = tf.concat((wconv, dconv), axis=1)
        cnn_output = tf.squeeze(cnn_output, axis=3)

        out_shape = cnn_output.get_shape()
        d = out_shape[1].value * out_shape[2].value
        fc_input = tf.reshape(cnn_output, [-1, d])

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

            output = tf.nn.xw_plus_b(fc_output, W, b, name="output_prob")

            self.prediction = tf.argmax(output, axis=1)

        with tf.name_scope("Loss"):
            one_hot_label = tf.one_hot(self.y_, depth=self.output_dim)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=tf.stop_gradient(one_hot_label))
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.y_)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.prediction, self.y_)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        global_step = tf.Variable(0, trainable=False)
        decayed_lr = tf.train.exponential_decay(config.lr, global_step, 3 * config.num_batch, 0.5, True)
        optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr)
        self.train_step = optimizer.minimize(self.loss, global_step=global_step)

    def __str__(self):
        name = "Wide and Deep CNN"
        embeddings = "Embedding Size : " + str(self.embedding)
        filter_num = "Number of Filters : " + str(self.config.filter_num)
        wide_layers = "Wide Conv Layers : " + str(self.wconv_layers)
        deep_layers = "Deep Conv Layers : " + str(self.conv_layers)

        return '\n'.join([name, embeddings, filter_num, wide_layers, deep_layers])