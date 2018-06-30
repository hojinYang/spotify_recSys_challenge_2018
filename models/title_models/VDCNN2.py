import tensorflow as tf
import numpy as np
import math

# weights initializers
he_normal = tf.keras.initializers.he_normal()
regularizer = tf.contrib.layers.l2_regularizer(1e-4)

def Convolutional_Block(inputs, shortcut, num_filters, name, is_training):
    # print("-"*20)
    # print("Convolutional Block", str(num_filters), name)
    # print("-"*20)
    with tf.variable_scope("conv_block_" + str(num_filters) + "_" + name):
        for i in range(2):
            with tf.variable_scope("conv1d_%s" % str(i)):
                filter_shape = [3, inputs.get_shape()[2], num_filters]
                W = tf.get_variable(name='W', shape=filter_shape,
                    initializer=he_normal,
                    regularizer=regularizer)
                inputs = tf.nn.conv1d(inputs, W, stride=1, padding="SAME")
                inputs = tf.layers.batch_normalization(inputs=inputs, momentum=0.997, epsilon=1e-5,
                                                center=True, scale=True, training=is_training)
                inputs = tf.nn.relu(inputs)
                # print("Conv1D:", inputs.get_shape())
    # print("-"*20)
    if shortcut is not None:
        # print("-"*5)
        # print("Optional Shortcut:", shortcut.get_shape())
        # print("-"*5)
        return inputs + shortcut
    return inputs

# Three types of downsampling methods described by paper
def downsampling(inputs, downsampling_type, name, optional_shortcut=False, shortcut=None):
    # k-maxpooling
    if downsampling_type=='k-maxpool':
        k = math.ceil(int(inputs.get_shape()[1]) / 2)
        pool = tf.nn.top_k(tf.transpose(inputs, [0,2,1]), k=k, name=name, sorted=False)[0]
        pool = tf.transpose(pool, [0,2,1])
    # Linear
    elif downsampling_type=='linear':
        pool = tf.layers.conv1d(inputs=inputs, filters=inputs.get_shape()[2], kernel_size=3,
                            strides=2, padding='same', use_bias=False)
    # Maxpooling
    else:
        pool = tf.layers.max_pooling1d(inputs=inputs, pool_size=3, strides=2, padding='same', name=name)
    if optional_shortcut:
        shortcut = tf.layers.conv1d(inputs=shortcut, filters=shortcut.get_shape()[2], kernel_size=1,
                            strides=2, padding='same', use_bias=False)
        # print("-"*5)
        # print("Optional Shortcut:", shortcut.get_shape())
        # print("-"*5)
        pool += shortcut
    pool = fixed_padding(inputs=pool)
    return tf.layers.conv1d(inputs=pool, filters=pool.get_shape()[2]*2, kernel_size=1,
                            strides=1, padding='valid', use_bias=False)

def fixed_padding(inputs, kernel_size=3):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [0, 0]])
    return padded_inputs

class VDCNN2():
    def __init__(self, config, depth=9, downsampling_type='maxpool', use_he_uniform=True, optional_shortcut=True):

        # Depth to No. Layers
        if depth == 9:
            self.num_layers = [2, 2, 2, 2]
        elif depth == 17:
            self.num_layers = [4, 4, 4, 4]
        elif depth == 29:
            self.num_layers = [10, 10, 4, 4]
        elif depth == 49:
            self.num_layers = [16, 16, 10, 6]
        else:
            raise ValueError('depth=%g is a not a valid setting!' % depth)

        self.config = config
        self.depth = depth
        self.lr = config.lr
        self.embedding_size = 16 # config.emb
        self.input_len = config.strmaxlen   # sequence_max_length=1024
        self.output_dim = config.output
        self.char_size = config.charsize
        self.downsampling_type = downsampling_type
        self.use_he_uniform = use_he_uniform
        self.optional_shortcut = optional_shortcut

        # input tensors
        self.x1 = tf.placeholder(tf.int64, [None, self.input_len], name="input_x")
        self.y_ = tf.placeholder(tf.int64, [None], name="input_y")
        # self.is_training = tf.placeholder(tf.bool)
        self.train = tf.placeholder(tf.bool) # True
        # self.train = True

        # Embedding Lookup 16
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if self.use_he_uniform:
                self.embedding_W = tf.get_variable(name='lookup_W', shape=[self.char_size, self.embedding_size], initializer=tf.keras.initializers.he_uniform())
            else:
                self.embedding_W = tf.Variable(tf.random_uniform([self.char_size, self.embedding_size], -1.0, 1.0), name="embedding_W")
            self.embedded_characters = tf.nn.embedding_lookup(self.embedding_W, self.x1)
            # print("-"*20)
            # print("Embedded Lookup:", self.embedded_characters.get_shape())
            # print("-"*20)

        self.layers = []

        # Temp(First) Conv Layer
        with tf.variable_scope("temp_conv") as scope:
            filter_shape = [3, self.embedding_size, 64]
            W = tf.get_variable(name='W_1', shape=filter_shape,
                initializer=he_normal,
                regularizer=regularizer)
            inputs = tf.nn.conv1d(self.embedded_characters, W, stride=1, padding="SAME")
            #inputs = tf.nn.relu(inputs)
        # print("Temp Conv", inputs.get_shape())
        self.layers.append(inputs)

        # Conv Block 64
        for i in range(self.num_layers[0]):
            if i < self.num_layers[0] - 1 and self.optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=64, is_training=self.train, name=str(i+1))
            self.layers.append(conv_block)
        pool1 = downsampling(self.layers[-1], downsampling_type=self.downsampling_type, name='pool1', optional_shortcut=self.optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool1)
        # print("Pooling:", pool1.get_shape())

        # Conv Block 128
        for i in range(self.num_layers[1]):
            if i < self.num_layers[1] - 1 and self.optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=128, is_training=self.train, name=str(i+1))
            self.layers.append(conv_block)
        pool2 = downsampling(self.layers[-1], downsampling_type=self.downsampling_type, name='pool2', optional_shortcut=self.optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool2)
        # print("Pooling:", pool2.get_shape())

        # Conv Block 256
        for i in range(self.num_layers[2]):
            if i < self.num_layers[2] - 1 and self.optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=256, is_training=self.train, name=str(i+1))
            self.layers.append(conv_block)
        pool3 = downsampling(self.layers[-1], downsampling_type=self.downsampling_type, name='pool3', optional_shortcut=self.optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool3)
        # print("Pooling:", pool3.get_shape())

        # Conv Block 512
        for i in range(self.num_layers[3]):
            if i < self.num_layers[3] - 1 and self.optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=512, is_training=self.train, name=str(i+1))
            self.layers.append(conv_block)

        # Extract 8 most features as mentioned in paper
        self.k_pooled = tf.nn.top_k(tf.transpose(self.layers[-1], [0,2,1]), k=8, name='k_pool', sorted=False)[0]
        # print("8-maxpooling:", self.k_pooled.get_shape())
        self.flatten = tf.reshape(self.k_pooled, (-1, 512*8))

        # fc1
        with tf.variable_scope('fc1'):
            w = tf.get_variable('w', [self.flatten.get_shape()[1], 2048], initializer=he_normal,
                regularizer=regularizer)
            b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(1.0))
            # out = tf.matmul(self.flatten, w) + b
            out = tf.nn.xw_plus_b(self.flatten, w, b)
            self.fc1 = tf.nn.relu(out)

        # fc2
        with tf.variable_scope('fc2'):
            w = tf.get_variable('w', [self.fc1.get_shape()[1], 2048], initializer=he_normal,
                regularizer=regularizer)
            b = tf.get_variable('b', [2048], initializer=tf.constant_initializer(1.0))
            # out = tf.matmul(self.fc1, w) + b
            out = tf.nn.xw_plus_b(self.fc1, w, b)
            self.fc2 = tf.nn.relu(out)

        # fc3
        with tf.variable_scope('fc3'):
            w = tf.get_variable('w', [self.fc2.get_shape()[1], self.output_dim], initializer=he_normal,
                regularizer=regularizer)
            b = tf.get_variable('b', [self.output_dim], initializer=tf.constant_initializer(1.0))
            self.fc3 = tf.nn.xw_plus_b(self.fc2, w, b)
            # self.fc3 = tf.matmul(self.fc2, w) + b


        self.prediction = tf.argmax(self.fc3, axis=1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("Loss"):
            one_hot_label = tf.one_hot(self.y_, depth=self.output_dim)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fc3, labels=tf.stop_gradient(one_hot_label))
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.fc3, labels=self.y_)
            self.loss = tf.reduce_mean(cross_entropy)

        # Accuracy
        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.prediction, self.y_)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        global_step = tf.Variable(0, trainable=False)
        decayed_lr = tf.train.exponential_decay(config.lr, global_step, 3 * config.num_batch, 0.5, True)
        optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grad, var = zip(*optimizer.compute_gradients(self.loss))
            grad = [
                None if g is None else tf.clip_by_norm(g, 3.0)
                for g in grad
            ]
            self.train_step = optimizer.apply_gradients(zip(grad, var))
        # self.train_step = optimizer.minimize(self.loss, global_step=global_step)


    def __str__(self):
        # depth=9, downsampling_type='maxpool', use_he_uniform=True, optional_shortcut=True
        name = "VDCNN2"
        embeddings = "Embedding Size : " + str(self.embedding_size)
        depth = "Depth : " + str(self.depth)
        layers = "Num Layers : " + str(self.num_layers)
        downsample = "Downsample Type: " + str(self.downsampling_type)
        shortcut = "Optional Shortcut : " + str(self.optional_shortcut)

        return '\n'.join([name, embeddings, depth, layers, downsample, shortcut])