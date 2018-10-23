from utils.data_reader import *
import utils.metrics as met
from models.DAEs import *
from models.title_get import get_model
import tensorflow as tf
import numpy as np
import os
import time
import argparse
import random
import datetime

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def log_write(conf, log):
    p = os.path.join(conf.dir, 'log.txt')
    with open(p, "a") as f:
        f.write(log)
        f.write('\n')
    if conf.verbose:
        print(log)


def eval(reader_test, conf, sess, model, model_title):
    # evaluation
    total_rprecision = 0
    total_ndcg = 0
    total_rsc = 0
    test_size = len(reader_test.playlists)
    while True:
        x_positions, test_seed, test_answer = reader_test.next_batch_test()
        predicted_matrix = sess.run(model.y_pred, feed_dict={model.x_positions: np.ones(len(x_positions)),
                                                             model.x_ones: x_ones,
                                                             model.keep_prob: 1.0, model.input_keep_prob: 1.0})

        predicted_matrix = predicted_matrix[:, :conf.n_tracks]
   
        for i in range(len(test_seed)):
            rprecision, ndcg, rsc = met.single_eval(predicted_matrix[i], test_seed[i], test_answer[i])
            total_rprecision += rprecision
            total_ndcg += ndcg
            total_rsc += rsc
        if reader_test.test_idx == 0:
            break

    total_rprecision /= test_size
    total_ndcg /= test_size
    total_rsc /= test_size
    return total_rprecision, total_ndcg, total_rsc


def show_result(rprecision, ndcg, rsc):
    return "rprecision: %f ndcg: %f rsc: %f" % (rprecision, ndcg, rsc)


def run(conf, only_testmode):
    reader = data_reader(data_dir=conf.data_dir, filename='train', batch_size=conf.batch)

    conf.n_tracks = reader.num_tracks
    conf.n_input = conf.n_tracks
    conf.n_output = conf.n_tracks

    kp_range = conf.input_kp
    test_seed = conf.test_seed

    readers_test = {}
    for seed in test_seed:
        readers_test[seed] = data_reader_test(data_dir=conf.data_dir, filename=seed,
                                              batch_size=conf.batch, test_num=conf.testsize)

    info = None
    model = None
    print(conf.n_input)

    if only_testmode:
        conf.initval = conf.save
    info = '[dae mode]'
    model = DAE(conf)

    info += ' start at ' + str(datetime.datetime.now())
    log_write(conf, '*'*10)
    log_write(conf, info)

    model.fit()
    sess = tf.Session()
    sess.run(model.init_op)
    saver = tf.train.Saver()
    
    epoch = 0
    max_eval = 0.0
    iter = 0
    loss = 0.0

    # if test mode is specified, just test the result and no training session.
    if only_testmode:
        log_write(conf, '<<only test mode>>')
        if conf.mode == 'title':
            saver.restore(sess, conf.save)

        for seed_num, reader_test in readers_test.items():
            log_write(conf, "seed num: " + seed_num)
            rprec, ndcg, rsc = eval(reader_test, conf, sess, model, model_title)
            r = show_result(rprec, ndcg, rsc)
            log_write(conf, r)
        return

    while True:
        start_idx = reader.train_idx
        trk_positions = reader.next_batch()
        end_idx = reader.train_idx

        input_kp = random.uniform(kp_range[0], kp_range[-1])

        _, l = sess.run([model.optimizer, model.cost],
                        feed_dict={model.x_positions: trk_positions, model.x_ones: np.ones(len(trk_positions)),
                                   model.y_positions: trk_positions, model.y_ones: np.ones(len(trk_positions)),
                                   model.keep_prob: conf.kp, model.input_keep_prob: input_kp})

        loss += l
        iter += 1

        if start_idx > end_idx or end_idx == 0:
            epoch += 1
            loss = loss / iter
            if epoch >= 0:
                log_write(conf, "epoch "+str(epoch))
                log_write(conf, "training loss: "+str(loss))

                for seed_num, reader_test in readers_test.items():
                    log_write(conf, "seed num: "+seed_num)
                    rprec, ndcg, rsc = eval(reader_test, conf, sess, model, model_title)
                    r = show_result(rprec, ndcg, rsc)
                    log_write(conf, r)

            loss = 0
            iter = 0
            if epoch == conf.epochs:
                break
