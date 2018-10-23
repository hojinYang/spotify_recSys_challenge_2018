"""
hojin yang
modified version of DAE for Spotify RecSys Challenge

"""

import os
import tensorflow as tf
import numpy as np
import pickle


class AE:
    def __init__(self, conf):
        self.save_dir = conf.save

        self.n_batch = conf.batch
        self.n_input = conf.n_input
        self.n_hidden = conf.n_hidden
        self.learning_rate = conf.lr

        self.x_positions = tf.placeholder(dtype=tf.int64,shape=[None,2])
        self.x_ones = tf.placeholder(dtype=tf.float32)

        self.y_positions = tf.placeholder(dtype=tf.int64,shape=[None,2])
        self.y_ones = tf.placeholder(dtype=tf.float32)
        
        self.keep_prob = tf.placeholder(tf.float32, shape=[])
        self.input_keep_prob = tf.placeholder(tf.float32, shape=[])

        with tf.device("/cpu:0"):
            x_sparse = tf.SparseTensor(indices=self.x_positions,
                                       values=self.x_ones,dense_shape=[self.n_batch,self.n_input])
            self.x = tf.sparse_tensor_to_dense(x_sparse, validate_indices=False)
            y_sparse = tf.SparseTensor(indices=self.y_positions,
                                       values=self.y_ones,dense_shape=[self.n_batch,self.n_input])
            self.y = tf.sparse_tensor_to_dense(y_sparse, validate_indices=False)

        x_dropout = tf.nn.dropout(self.x, keep_prob=self.input_keep_prob)
        self.reduce_sum = tf.reduce_sum(x_dropout, 1, keepdims=True)
        self.x_dropout = tf.divide(x_dropout, self.reduce_sum + 1e-10)

        self.y_pred = None
        self.cost = None
        self.optimizer = None
        self.init_op = None

        self.weights = {}
        self.biases = {}
        self.d_params = []

    def init_weight(self):
        if self.initval_dir == 'NULL':
            self.weights['encoder_h'] = tf.get_variable("encoder_h", shape=[self.n_input, self.n_hidden],
                                                        initializer=tf.contrib.layers.xavier_initializer())
            self.weights['decoder_h'] = tf.get_variable("decoder_h", shape=[self.n_input, self.n_hidden],
                                                        initializer=tf.contrib.layers.xavier_initializer())
            self.biases['encoder_b'] = tf.get_variable(name="encoder_b", shape=[self.n_hidden],
                                                       initializer=tf.zeros_initializer())
            self.biases['decoder_b'] = tf.get_variable(name="decoder_b", shape=[self.n_input],
                                                       initializer=tf.zeros_initializer())
        else:
            with open(self.initval_dir, 'rb') as f:
                emb = pickle.load(f)
            self.weights['encoder_h'] = tf.get_variable("encoder_h", initializer=tf.constant(emb[0]))
            self.weights['decoder_h'] = tf.get_variable("decoder_h", initializer=tf.constant(emb[1]))
            self.biases['encoder_b'] = tf.get_variable(name="encoder_b", initializer=tf.constant(emb[2]))
            self.biases['decoder_b'] = tf.get_variable(name="decoder_b", initializer=tf.constant(emb[3]))

        self.d_params = [self.weights['encoder_h'], self.weights['decoder_h'],
                         self.biases['encoder_b'], self.biases['decoder_b']]

    # Building the encoder
    def encoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1         
        layer = tf.add(tf.matmul(x, self.weights['encoder_h']), self.biases['encoder_b'])
        layer = tf.nn.sigmoid(layer)
        layer = tf.nn.dropout(layer, self.keep_prob)

        return layer

    # Building the decoder
    def decoder(self, x):
        # Decoder Hidden layer with sigmoid activation #1
        layer = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(self.weights['decoder_h'])),
                                     self.biases['decoder_b']))
        return layer

    def fit(self):
        # Construct model
        with tf.device("/cpu:0"):  #CPU
            self.init_weight()

        encoder_op = self.encoder(self.x_dropout)
        with tf.device("/gpu:1"):  #GPU1
            self.y_pred = self.decoder(encoder_op)
            
        # Define loss and optimizer, minimize the squared error
        with tf.device("/gpu:1"): ##SHOULD BE GPU1
            L = -tf.reduce_sum(self.y*tf.log(self.y_pred+1e-10) + 
                               0.55*(1 - self.y)* tf.log(1 - self.y_pred+1e-10),axis = 1)
            self.cost = tf.reduce_mean(L) + self.reg_lambda * l2

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        # Initialize the variables (i.e. assign their default value)
        self.init_op = tf.global_variables_initializer()
        
    def save_model(self, sess):
        param = sess.run(self.d_params)
        output = open(self.save_dir, 'wb')
        pickle.dump(param, output)
        output.close()


