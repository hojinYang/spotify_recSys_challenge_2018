# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class VDCNN():
    def __init__(self, config):
        self.config = config
        self.embedding = config.emb
        self.input_len = config.strmaxlen
        self.output_dim = config.output
        self.char_size = config.charsize
        self.lr = config.lr
        self.K = 8

        with tf.name_scope("Input-Layer"):
            # Input
            self.x1 = tf.placeholder(tf.int64, shape=[None, self.input_len], name="input_x")
            self.y_ = tf.placeholder(tf.int64, shape=[None], name="output_x")
            self.train = True
            # self.train = tf.placeholder(tf.bool, name='Train')
            if self.embedding != 0:
                stdv = 1 / np.sqrt(self.embedding)
                init = tf.contrib.layers.xavier_initializer(uniform=False)
                embedding_matrix = tf.get_variable('char_embedding', [self.char_size, self.embedding], initializer=init)
                # embedding_matrix = tf.Variable(tf.random_uniform(shape=[self.char_size, self.embedding], minval=-stdv, maxval=stdv),dtype=tf.float32, name="Embedding_Weight")

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
                emb_x = tf.nn.embedding_lookup(one_hot, self.x1)
            else:
                emb_x = tf.nn.embedding_lookup(embedding_matrix, self.x1)
            # Embedding Weight 후 (Batch, input_len, emb_dim)

            cnn_x = tf.expand_dims(emb_x, -1)

        def Conv(input, filter_shape, strides, train, scope):
            # stdv = 1 / np.sqrt(filter_shape[1] * filter_shape[2])
            # norm = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
            # norm = tf.contrib.layers.xavier_initializer(uniform=False)

            norm = tf.random_normal_initializer(stddev=0.05)
            const = tf.constant_initializer(0.0)
            with tf.variable_scope(scope):
                filter_1 = tf.get_variable('filter1', filter_shape, initializer=norm)
                conv = tf.nn.conv2d(input, filter_1, strides=strides, padding="SAME")
                batch_norm = tf.layers.batch_normalization(conv, trainable=train, name=scope + "BN")
                return batch_norm

        def Convolutional_Block(input, filter_num, train, scope):
            # stdv = 1 / np.sqrt(filter_num)
            # norm = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
            # norm = tf.contrib.layers.xavier_initializer(uniform=False)

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
            # norm = tf.random_uniform_initializer(minval=-stddev, maxval=stddev)
            # norm = tf.contrib.layers.xavier_initializer(uniform=False)

            norm = tf.random_normal_initializer(stddev=stddev)
            const = tf.constant_initializer(0.0)
            with tf.variable_scope(scope or 'linear'):
                w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
                b = tf.get_variable('b', [output_dim], initializer=const)
                l2_loss = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
                # return tf.matmul(input, w) + b, l2_loss
                return tf.matmul(tf.nn.dropout(input, 0.8), w) + b, l2_loss

        with tf.name_scope("VDCNN-Layer-0"):
            embedding_size = self.embedding if self.embedding != 0 else self.char_size
            filter_shape0 = [3, embedding_size, 1, 64]
            strides0 = [1, 1, embedding_size, 1]
            h0 = Conv(cnn_x, filter_shape0, strides0, train=self.train, scope='VDCNN-Layer-0')

        with tf.name_scope("VDCNN-Layer-1-8"):
            h1 = Convolutional_Block(h0, 64, self.train, 'layer_1-2')
            h2 = Convolutional_Block(h1, 128, self.train, 'layer_3-4')
            h3 = Convolutional_Block(h2, 256, self.train, 'layer_5-6')
            h4 = Convolutional_Block(h3, 512, self.train, 'layer_7-8')
            h5 = tf.transpose(h4, [0, 3, 2, 1])

            pooled = tf.nn.top_k(h5, k=self.K, name='k-maxpooling')  # ?, 10, 1, 512
            # flat_size = self.K * h5.shape[1].value
            flat_size = 8*512
            h6 = tf.reshape(pooled[0], (-1, 512 * 8))

        l2_loss = tf.constant(0.0)

        with tf.name_scope("FC-Layers"):
            fc_1, fc_1_loss = linear(h6, 2048, scope='FC1', stddev=0.1)
            fc_2, fc_2_loss = linear(tf.nn.relu(fc_1), 2048, scope='FC2', stddev=0.1)
            self.fc_3, fc_3_loss = linear(tf.nn.relu(fc_2), self.output_dim, scope='FC3', stddev=0.1)
            l2_loss += fc_1_loss + fc_2_loss + fc_3_loss

        self.prediction = tf.argmax(self.fc_3, 1, name="predictions")

        with tf.name_scope("Loss"):
            one_hot_label = tf.one_hot(self.y_, depth=self.output_dim)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fc_3, labels=tf.stop_gradient(one_hot_label))
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.fc_3, labels=self.y_)
            self.loss = tf.reduce_mean(cross_entropy)

        global_step = tf.Variable(0, trainable=False)
        decayed_lr = tf.train.exponential_decay(config.lr, global_step, 3 * config.num_batch, 0.5, True)
        optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr)
        self.train_step = optimizer.minimize(self.loss, global_step=global_step)

        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.prediction, self.y_)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def fit(self):
        pass