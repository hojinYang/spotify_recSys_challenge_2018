# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class Model():
    def __init__(self, config):
        # input_len = config.strmaxlen
        # char_size = config.charsize
        # output_dim = 1
        # lr = config.lr
        train = True if config.mode == 'train' else False

        config = config
        embedding = config.emb
        input_len = config.strmaxlen
        output_dim = config.output
        char_size = config.charsize
        lr = config.lr
        K = 8

        ### Modeling
        ##########################################################
        with tf.name_scope("Input-Layer"):
            # Input
            self.x1 = tf.placeholder(tf.int64, shape=[None, input_len], name="input_x")
            self.y_ = tf.placeholder(tf.float32, shape=[None], name="output_x")
            # keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            if config.emb != 0:
                stdv = 1 / np.sqrt(config.emb)
                embedding_matrix = tf.Variable(tf.random_uniform(shape=[char_size, config.emb], minval=-stdv, maxval=stdv),dtype=tf.float32, name="Embedding_Weight")

        # EMBEDDING LAYERS
        with tf.name_scope("Embedding-Layer"):
            # one hot encoding of zero vector, 0, 1, 2, ... , char_size - 1
            one_hot = tf.concat(
                [
                    tf.zeros([1, char_size]),
                    tf.one_hot(list(range(char_size)), char_size, 1.0, 0.0)
                ],
                0
            )
            # CNN (tf.conv2d) : input = [batch, in_height, in_width, in_channels]
            # NLP embedding only has 1 channels
            # shape = (Batch, input_len, char_size, 1)
            if config.emb == 0:
                emb_x = tf.nn.embedding_lookup(one_hot, self.x1)
            else:
                emb_x = tf.nn.embedding_lookup(embedding_matrix, self.x1)
            # Embedding Weight 후 (Batch, input_len, emb_dim)

            cnn_x = tf.expand_dims(emb_x, -1)

        check_point = 0

        def Conv(input, filter_shape, strides, train, scope):
            norm = tf.random_normal_initializer(stddev=0.05)
            const = tf.constant_initializer(0.0)
            with tf.variable_scope(scope):
                filter_1 = tf.get_variable('filter1', filter_shape, initializer=norm)
                conv = tf.nn.conv2d(input, filter_1, strides=strides, padding="SAME")
                batch_norm = tf.layers.batch_normalization(conv, trainable=train, name=scope + "BN")
                return batch_norm

        def Convolutional_Block(input, filter_num, train, scope):
            norm = tf.random_normal_initializer(stddev=0.05)
            const = tf.constant_initializer(0.0)
            filter_shape1 = [3, 1, input.get_shape()[3], filter_num]

            with tf.variable_scope(scope):
                filter_1 = tf.get_variable('filter1', filter_shape1, initializer=norm)
                conv1 = tf.nn.conv2d(input, filter_1, strides=[1, 1, filter_shape1[1], 1], padding="SAME")
                batch_normal1 = tf.layers.batch_normalization(conv1, trainable=train, name=scope + "BN1")
                filter_shape2 = [3, 1, batch_normal1.get_shape()[3], filter_num]
                filter_2 = tf.get_variable('filter2', filter_shape2, initializer=norm)
                conv2 = tf.nn.conv2d(tf.nn.relu(batch_normal1), filter_2, strides=[1, 1, filter_shape2[1], 1],
                                     padding="SAME")
                batch_normal2 = tf.layers.batch_normalization(conv2, trainable=train, name=scope + "BN2")
                pooled = tf.nn.max_pool(tf.nn.relu(batch_normal2), ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1],
                                        padding='SAME', name="pool1")
                return pooled

        def linear(input, output_dim, scope=None, stddev=0.1):
            norm = tf.random_normal_initializer(stddev=stddev)
            const = tf.constant_initializer(0.0)
            with tf.variable_scope(scope or 'linear'):
                w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
                b = tf.get_variable('b', [output_dim], initializer=const)
                l2_loss = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
                return tf.matmul(input, w) + b, l2_loss

        with tf.name_scope("VDCNN-Layer-0"):
            embedding_size = config.emb if config.emb != 0 else config.charsize
            filter_shape0 = [3, embedding_size, 1, 64]
            strides0 = [1, 1, embedding_size, 1]
            h0 = Conv(cnn_x, filter_shape0, strides0, train=train, scope='VDCNN-Layer-0')

        with tf.name_scope("VDCNN-Layer-1-8"):
            h1 = Convolutional_Block(h0, 64, train, 'layer_1-2')
            h2 = Convolutional_Block(h1, 128, train, 'layer_3-4')
            h3 = Convolutional_Block(h2, 256, train, 'layer_5-6')
            h4 = Convolutional_Block(h3, 512, train, 'layer_7-8')
            h5 = tf.transpose(h4, [0, 3, 2, 1])

            pooled = tf.nn.top_k(h5, k=8, name='k-maxpooling')
            h6 = tf.reshape(pooled[0], (-1, 512 * 8))
        l2_loss = tf.constant(0.0)
        with tf.name_scope("FC-Layers"):
            fc_1, fc_1_loss = linear(h6, 2048, scope='FC1', stddev=0.1)
            fc_2, fc_2_loss = linear(tf.nn.relu(fc_1), 2048, scope='FC2', stddev=0.1)
            fc_3, fc_3_loss = linear(tf.nn.relu(fc_2), output_dim, scope='FC3', stddev=0.1)
            l2_loss += fc_1_loss + fc_2_loss + fc_3_loss

            # output_rating = tf.sigmoid(fc_3) * 9 + 1
            self.output_sigmoid = tf.reshape(fc_3, [-1])

        with tf.name_scope("Loss"):
            self.loss = tf.losses.mean_squared_error(predictions=self.output_sigmoid, labels=self.y_)

        global_step = tf.Variable(0, trainable=False)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_step = optimizer.minimize(self.loss, global_step=global_step)
    def fit(self):
        pass