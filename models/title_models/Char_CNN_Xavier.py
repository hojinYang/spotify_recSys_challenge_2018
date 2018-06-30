# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

class Char_CNN():
    def __init__(self, config, conv_layers):
        self.config = config
        self.embedding = config.emb
        self.input_len = config.strmaxlen
        self.output_dim = config.output
        self.char_size = config.charsize
        self.conv_layers = conv_layers

        """
        [Best Config]
        batch=100, batch_norm=False, 
        bi=True, charsize=2515, 
        dropout=False, emb=0, epochs=30, 
        iteration='0', 
        kernel=64, mode='train', 
        output=1, 
        pause=0, 
        strmaxlen=420
        conv_layers = [[64, 3, -1], [64, 3, 3], [64, 3, -1], [64, 3, 3], [64, 3, -1], [64, 3, 3]]
        """
        with tf.name_scope("Input-Layer"):
            # Input
            self.x1 = tf.placeholder(tf.int64, shape=[None, self.input_len], name="input_x")
            self.y_ = tf.placeholder(tf.int64, shape=[None], name="output_x")
            keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            if self.embedding != 0:
                init = tf.contrib.layers.xavier_initializer(uniform=False)
                embedding_matrix = tf.get_variable('char_embedding', [self.char_size, self.embedding], initializer=init)

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
            if self.embedding == 0:
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

                W = tf.get_variable(name="Conv_W" + str(i), shape=filter_shape, dtype=tf.float32,
                                    initializer=tf.keras.initializers.he_normal())
                b = tf.Variable(tf.random_uniform(shape=[conv_info[0]], minval=0, maxval=0), dtype=tf.float32,
                                name='Conv_b')

                conv = tf.nn.conv2d(cnn_x, W, [1, 1, 1, 1], "VALID", name="conv")

            with tf.name_scope("Non-Linear"):
                cnn_x = tf.nn.bias_add(conv, b)
                cnn_x = tf.nn.crelu(cnn_x) #relu6가 가장 좋았음. 811
                print("crelu")
            if conv_info[-1] != -1:
                with tf.name_scope("Max-Polling"):
                    pool_shape = [1, conv_info[-1], 1, 1]
                    pool = tf.nn.max_pool(cnn_x, ksize=pool_shape, strides=pool_shape, padding="VALID")
                    cnn_x = tf.transpose(pool, [0, 1, 3, 2])
            else:
                cnn_x = tf.transpose(cnn_x, [0, 1, 3, 2])
        cnn_output = tf.squeeze(cnn_x, axis=3)

        d = cnn_output.shape[1].value * cnn_output.shape[2].value
        cnn_output = tf.reshape(cnn_output, [-1, d])

        with tf.name_scope("Output-Layer"):
            W = tf.get_variable(name="Output_W", shape=[d, self.output_dim], dtype=tf.float32,
                                initializer=tf.keras.initializers.glorot_normal())
            b = tf.Variable(tf.random_uniform([self.output_dim], minval=0, maxval=0), dtype=tf.float32,
                            name="Output_b")

            output = tf.nn.xw_plus_b(cnn_output, W, b, name="output_prob")
            softmax = tf.nn.softmax(output, axis=1)
            self.prediction = tf.argmax(softmax, axis=1)


        with tf.name_scope("Loss"):
            one_hot_label = tf.one_hot(self.y_, depth=self.output_dim)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=tf.stop_gradient(one_hot_label))
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_output, labels=self.y_)
            self.loss = tf.reduce_mean(cross_entropy)


        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.prediction, self.y_)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
        self.train_step = optimizer.minimize(self.loss, global_step=global_step)

    def __str__(self):
        name = "Char_CNN_Xavier"
        embeddings = "Embedding Size : " + str(self.embedding)
        filter_num = "Number of Filters : " + str(self.config.filter_num)
        layers = "Conv Layers : " + str(self.conv_layers)

        return '\n'.join([name, embeddings, filter_num, layers])