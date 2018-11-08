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
<<<<<<< HEAD
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
=======
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
>>>>>>> e699f7b47d3f20b6e90a780eb2af4224ea75800b



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
    if -1 in conf.firstN:
        reader = data_reader(data_dir=conf.data_dir, filename='train', batch_size=conf.batch)
    else:
        reader = data_reader_firstN(data_dir=conf.data_dir, filename='train',
                                    batch_size=conf.batch, from_to=conf.firstN)

    conf.class_divpnt = reader.class_divpnt
    conf.n_tracks = reader.num_tracks
<<<<<<< HEAD
    conf.n_input = reader.num_items
    conf.n_output = reader.num_items
    conf.charsize = reader.num_char
    conf.strmaxlen = reader.max_title_len
=======

    conf.n_input = conf.n_tracks
    conf.n_output = conf.n_tracks
>>>>>>> e699f7b47d3f20b6e90a780eb2af4224ea75800b

    kp_range = conf.input_kp
    test_seed = conf.test_seed
    update_seed = conf.update_seed

    readers_test = {}
    for seed in test_seed:
        readers_test[seed] = data_reader_test(data_dir=conf.data_dir, filename=seed,
                                              batch_size=conf.batch, test_num=conf.testsize)

    print(conf.n_input)

<<<<<<< HEAD
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
=======
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
>>>>>>> e699f7b47d3f20b6e90a780eb2af4224ea75800b

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
        trk_positions, art_positions, y_positions, titles, trk_val, art_val = reader.next_batch()
        end_idx = reader.train_idx

        input_kp = random.uniform(kp_range[0], kp_range[-1])

<<<<<<< HEAD
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
=======
        _, l = sess.run([model.optimizer, model.cost],
                        feed_dict={model.x_positions: trk_positions, model.x_ones: np.ones(len(trk_positions)),
                                   model.y_positions: trk_positions, model.y_ones: np.ones(len(trk_positions)),
                                   model.keep_prob: conf.hidden_kp, model.input_keep_prob: input_kp})
>>>>>>> e699f7b47d3f20b6e90a780eb2af4224ea75800b

        loss += l
        iter += 1


        if start_idx > end_idx or end_idx == 0:

            epoch += 1
            loss = loss / iter
            if epoch >= 0:
<<<<<<< HEAD
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
=======
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


>>>>>>> e699f7b47d3f20b6e90a780eb2af4224ea75800b
            loss = 0
            iter = 0
            if epoch == conf.epochs:
                break
