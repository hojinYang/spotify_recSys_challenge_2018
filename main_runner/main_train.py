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
        predicted_matrix = None
        x_positions, test_seed, test_answer, titles, x_ones = reader_test.next_batch_test()
        if conf.mode in ['pretrain', 'dae']:
            predicted_matrix = sess.run(model.y_pred, feed_dict={model.x_positions: x_positions,
                                                                 model.x_ones: x_ones,
                                                                 model.keep_prob: 1.0, model.input_keep_prob: 1.0})
        elif conf.mode == 'title':
            len_titles = len(titles)
            if len_titles < conf.batch:
                zeros = [-1] * conf.strmaxlen
                titles = titles + [zeros] * (conf.batch - len_titles)
            predicted_matrix = sess.run(model.y_pred, feed_dict={model.x_positions: x_positions,
                                                                 model.x_ones: x_ones,
                                                                 model_title.titles: titles,
                                                                 model.keep_prob: 1.0, model_title.keep_prob: 1.0,
                                                                 model.input_keep_prob: 1.0,
                                                                 model.titles_use: [[1]] * conf.batch})

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
    if -1 in conf.firstN:
        reader = data_reader(data_dir=conf.data_dir, filename='train', batch_size=conf.batch)
    else:
        reader = data_reader_firstN(data_dir=conf.data_dir, filename='train',
                                    batch_size=conf.batch, from_to=conf.firstN)

    conf.n_tracks = reader.num_tracks
    conf.n_input = reader.num_items
    conf.n_output = reader.num_items
    conf.charsize = reader.num_char
    conf.strmaxlen = reader.max_title_len

    kp_range = conf.input_kp
    test_seed = conf.test_seed
    update_seed = conf.update_seed

    readers_test = {}
    for seed in test_seed:
        readers_test[seed] = data_reader_test(data_dir=conf.data_dir, filename=seed,
                                              batch_size=conf.batch, test_num=conf.testsize)

    info = None
    model = None
    print(conf.n_input)

    model_title = None
    if conf.mode == 'pretrain':
        info = '[pretrain mode]'
        model = DAE_tied(conf)
    elif conf.mode == 'dae':
        if only_testmode:
            conf.initval = conf.save
        info = '[dae mode]'
        model = DAE(conf)
    elif conf.mode == 'title':
        info = '[title mode]'
        model_title = get_model(conf)
        model = DAE_title(conf, model_title.output)

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
        trk_positions, art_positions, y_positions, titles, trk_val, art_val = reader.next_batch()
        end_idx = reader.train_idx

        input_kp = random.uniform(kp_range[0], kp_range[-1])

        if conf.mode in ['pretrain', 'dae']:
            rand_int = np.random.randint(2)
            if rand_int == 0:
                _, l = sess.run([model.optimizer, model.cost],
                                feed_dict={model.x_positions: trk_positions, model.x_ones: trk_val,
                                           model.y_positions: y_positions, model.y_ones: np.ones(len(y_positions)),
                                           model.keep_prob: conf.kp, model.input_keep_prob: input_kp})

            elif rand_int == 1:
                _, l = sess.run([model.optimizer, model.cost],
                                feed_dict={model.x_positions: art_positions, model.x_ones: art_val,
                                           model.y_positions: y_positions, model.y_ones: np.ones(len(y_positions)),
                                           model.keep_prob: conf.kp, model.input_keep_prob: input_kp})
        elif conf.mode == 'title':
            _, l = sess.run([model.optimizer, model.cost],
                            feed_dict={model.x_positions: y_positions, model.x_ones: np.ones(len(y_positions)),
                                       model.y_positions: y_positions, model.y_ones: np.ones(len(y_positions)),
                                       model_title.titles: titles,
                                       model.keep_prob: conf.kp, model_title.keep_prob: conf.title_kp,
                                       model.input_keep_prob: input_kp,
                                       model.titles_use: [[1]] * conf.batch})

        loss += l
        iter += 1

        if start_idx > end_idx or end_idx == 0:
            epoch += 1
            loss = loss / iter
            if epoch >= 0:
                log_write(conf, "epoch "+str(epoch))
                log_write(conf, "training loss: "+str(loss))
                cur_eval = 0
                for seed_num, reader_test in readers_test.items():
                    log_write(conf, "seed num: "+seed_num)
                    rprec, ndcg, rsc = eval(reader_test, conf, sess, model, model_title)
                    r = show_result(rprec, ndcg, rsc)
                    log_write(conf, r)
                    if seed_num in update_seed:
                        cur_eval += rprec

                if cur_eval >= max_eval:
                    if conf.mode in ['pretrain', 'dae']:
                        model.save_model(sess)
                    elif conf.mode == 'title':
                        saver.save(sess, conf.save)
                    max_eval = cur_eval
                    log_write(conf, "The highest score is updated. Parameters are saved")
            loss = 0
            iter = 0
            if epoch == conf.epochs:
                break
