from utils.data_reader import *
import utils.metrics as met
from models.Autoencoder import *
import tensorflow as tf
import numpy as np
import os
import time
import argparse
import random
import datetime
import re

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def log_write(dir, test, log):
    test = test+'.txt'
    p = os.path.join(dir, test)
    with open(p, "a") as f:
        f.write(log)
        f.write('\n')
    print(log)

def get_row_wise_sqrsum(mat):
    s = np.sum(np.square(mat), axis = 1)
    return s

def get_avg_by_cls(item_list, class_divpnt):
    avg_by_cls = []
    start = 0
    for end in class_divpnt:
        cls = item_list[start:end]
        avg_by_cls.append(sum(cls)/len(cls))
        start = end
    cls = item_list[start:]
    avg_by_cls.append(sum(cls) / len(cls))
    return avg_by_cls


def get_encdec_avg_by_cls(conf, sess, model):
    encoder_sqrsum, decoder_sqrsum = sess.run([model.encoder_sqrsum, model.decoder_sqrsum])
    encoder_avg_by_cls = get_avg_by_cls(encoder_sqrsum.tolist(), conf.class_divpnt)
    decoder_avg_by_cls = get_avg_by_cls(decoder_sqrsum.tolist(), conf.class_divpnt)
    return encoder_avg_by_cls, decoder_avg_by_cls


def eval(reader_test, conf, sess, model):
    # evaluation
    ssq = 0
    total_rprecision = 0
    hit_by_cls_mat = []
    cand_cls_dist_mat = []
    encoder_grad_sqrsum_by_cls = []
    decoder_grad_sqrsum_by_cls = []

    total_hidden_sqrsum = 0
    # total_ndcg = 0
    # total_rsc = 0
    test_size = len(reader_test.playlists)

    while True:
        x_positions, test_seed, test_answer, test_answer_cls, answers_for_grad = reader_test.next_batch_test()

        predicted_matrix, hidden_mat, encoder_grad_sqrsum, decoder_grad_sqrsum = \
            sess.run([model.y_pred, model.hidden, model.encoder_grad_sqrsum, model.decoder_grad_sqrsum],
                     feed_dict={model.x_positions: x_positions,
                                model.x_ones: np.ones(len(x_positions)),
                                model.y_positions: answers_for_grad,
                                model.y_ones: np.ones(len(answers_for_grad)),
                                model.keep_prob: 1.0, model.input_keep_prob: 1.0})

        hidden_sqrsum = get_row_wise_sqrsum(hidden_mat)[:len(test_seed)]
        e_g = get_avg_by_cls(encoder_grad_sqrsum, conf.class_divpnt)
        d_g = get_avg_by_cls(decoder_grad_sqrsum, conf.class_divpnt)
        encoder_grad_sqrsum_by_cls.append(e_g)
        decoder_grad_sqrsum_by_cls.append(d_g)



        total_hidden_sqrsum += np.sum(hidden_sqrsum)

        # ssq += np.sum(grad[0]**2)
        predicted_matrix = predicted_matrix[:, :conf.n_tracks]
   
        for i in range(len(test_seed)):
            rprecision, hit_by_cls, cand_cls_dist = met.single_eval(predicted_matrix[i], test_seed[i], test_answer[i],
                                                     test_answer_cls[i], conf.class_divpnt)
            total_rprecision += rprecision
            hit_by_cls_mat.append(hit_by_cls)
            cand_cls_dist_mat.append(cand_cls_dist)
            # print(hit_by_cls_mat)
            # total_ndcg += ndcg
            # total_rsc += rsc
        if reader_test.test_idx == 0:
            break

    total_rprecision /= test_size
    total_hidden_sqrsum /= test_size

    hit_by_cls_mat = np.matrix(hit_by_cls_mat)
    hr_by_cls = hit_by_cls_mat.mean(axis=0).tolist()[0]

    cand_cls_dist_mat = np.matrix(cand_cls_dist_mat)
    cand_cls_dist = cand_cls_dist_mat.mean(axis=0).tolist()[0]

    encoder_grad_sqrsum_by_cls = np.matrix(encoder_grad_sqrsum_by_cls)
    encoder_grad_sqrsum_by_cls = encoder_grad_sqrsum_by_cls.mean(axis=0).tolist()[0]

    decoder_grad_sqrsum_by_cls = np.matrix(decoder_grad_sqrsum_by_cls)
    decoder_grad_sqrsum_by_cls = decoder_grad_sqrsum_by_cls.mean(axis=0).tolist()[0]



    # total_ndcg /= test_size
    # total_rsc /= test_size
    # print(ssq)
    return total_rprecision, hr_by_cls, cand_cls_dist, total_hidden_sqrsum, \
           encoder_grad_sqrsum_by_cls, decoder_grad_sqrsum_by_cls


def show_result(rprecision, ndcg, rsc):
    return "rprecision: %f ndcg: %f rsc: %f" % (rprecision, ndcg, rsc)


def run(conf, only_testmode):
    reader = data_reader(data_dir=conf.data_dir, filename='train', batch_size=conf.batch)

    conf.class_divpnt = reader.class_divpnt
    conf.n_tracks = reader.num_tracks

    conf.n_input = conf.n_tracks
    conf.n_output = conf.n_tracks

    kp_range = conf.input_kp
    test_seed = conf.test_seed

    readers_test = {}
    for seed in test_seed:
        readers_test[seed] = data_reader_test(data_dir=conf.data_dir, filename=seed,
                                              batch_size=conf.batch, test_num=conf.testsize)

    print(conf.n_input)

    if only_testmode:
        conf.initval = conf.save
    model = AE(conf)

    # info = ' start at ' + str(datetime.datetime.now())
    dir = str(datetime.datetime.now())
    dir_list = re.findall('\d+', dir)
    dir = ""
    for i in dir_list:
        dir+=i
    dir = os.path.join(conf.dir, dir)
    os.mkdir(dir)
    # log_write(conf, '*'*10)
    # log_write(conf, info)

    model.fit()
    sess = tf.Session()
    sess.run(model.init_op)
    saver = tf.train.Saver()
    
    epoch = 0
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
                                   model.keep_prob: conf.hidden_kp, model.input_keep_prob: input_kp})

        loss += l
        iter += 1


        if start_idx > end_idx or end_idx == 0:

            epoch += 1
            loss = loss / iter
            if epoch >= 0:
                r = "%d %f"%(epoch, loss)
                log_write(dir, "loss", r)

                encoder_avg_by_cls, decoder_avg_by_cls = get_encdec_avg_by_cls(conf, sess, model)
                r = "%d " % epoch
                for i in encoder_avg_by_cls:
                    r += "%f " % i
                log_write(dir, "encoder_avg_by_cls", r)

                r = "%d " % epoch
                for i in decoder_avg_by_cls:
                    r += "%f " % i
                log_write(dir, "decoder_avg_by_cls", r)


                for seed_num, reader_test in readers_test.items():
                    r = "%d " %epoch
                    # log_write(conf, "seed num: "+seed_num)
                    rprec, hr_by_cls, cand_cls_dist, total_hidden_sqrsum, \
                    encoder_grad_sqrsum_by_cls, decoder_grad_sqrsum_by_cls= eval(reader_test, conf, sess, model)
                    # r = show_result(rprec, ndcg, rsc)
                    r += "%f " %rprec

                    for i in hr_by_cls:
                        r += "%f " %i
                    log_write(dir, seed_num, r)

                    r = "%d " %epoch
                    for i in cand_cls_dist:
                        r += "%f " % i
                    log_write(dir, seed_num+"-cand_cls_dist", r)

                    r = "%d " % epoch
                    for i in encoder_grad_sqrsum_by_cls:
                        r += "%f " % i
                    log_write(dir, seed_num + "-encoder_grad_sqrsum_by_cls", r)

                    r = "%d " % epoch
                    for i in decoder_grad_sqrsum_by_cls:
                        r += "%f " % i
                    log_write(dir, seed_num + "-decoder_grad_sqrsum_by_cls", r)

                    r = "%d %f" % (epoch, total_hidden_sqrsum)
                    log_write(dir, seed_num + "-hidden_sqrsum", r)


            loss = 0
            iter = 0
            if epoch == conf.epochs:
                break
