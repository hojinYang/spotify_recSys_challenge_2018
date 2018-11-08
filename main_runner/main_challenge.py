from utils.data_reader import data_reader, data_reader_challenge
import utils.metrics as met
from models.DAEs import *
from models.title_get import get_model
import tensorflow as tf
import numpy as np
import os
import time
import argparse
import random
import pandas as pd
import datetime
import pickle

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def log_write(conf, log):
    p = os.path.join(conf.dir, 'log.txt')
    with open(p, "a") as f:
        f.write(log)
        f.write('\n')
    if conf.verbose:
        print(log)


def cand_generate(scores, seed, id2uri):
    cand_uri = []
    cand = np.argsort(-1*scores)
    cand = cand.tolist()

    for i in seed:
        try:
            cand.remove(i)
        except:
            pass
    cand = cand[:500]
    for i in cand:
        uri = id2uri[str(i)]
        uri = 'spotify:track:'+uri
        cand_uri.append(uri)
    return cand_uri
        
        
def run(conf):
    ## set input len trk len

    reader_challenge = data_reader_challenge(data_dir=conf.data_dir, filename=conf.challenge_data, batch_size=conf.batch)

    conf.n_tracks = reader_challenge.num_tracks
    conf.n_input = reader_challenge.num_items
    conf.n_output = reader_challenge.num_items
    conf.charsize = reader_challenge.num_char
    conf.strmaxlen = reader_challenge.max_title_len

    print(conf.n_input)

    info = '[challenge mode]'
    model_title = get_model(conf)
    model = DAE_title(conf, model_title.output)

    info += ' start at ' + str(datetime.datetime.now())
    log_write(conf, '*'*10)
    log_write(conf, info)

    model.fit()
    sess = tf.Session()
    sess.run(model.init_op)
    saver = tf.train.Saver()
    saver.restore(sess, conf.save)
    # total_cands = [['team_info', 'main', 'hello world!', 'hojinyang7@skku.edu']]
    total_cands = []
    while True:
        x_positions, seed, titles, titles_exist, pid, x_ones = reader_challenge.next_batch()
        len_titles = len(titles)
        if len_titles < conf.batch:
            zeros = [-1] * conf.strmaxlen
            titles = titles + [zeros] * (conf.batch - len_titles)
            titles_exist = titles_exist + [[0]] * (conf.batch - len_titles)

        predicted_matrix = sess.run(model.y_pred, feed_dict={model.x_positions: x_positions,
                                                             model.x_ones: x_ones,
                                                             model_title.titles: titles,
                                                             model.keep_prob: 1.0, model_title.keep_prob: 1.0,
                                                             model.input_keep_prob: 1.0,
                                                             model.titles_use: titles_exist})
        
        predicted_matrix = predicted_matrix[:, :conf.n_tracks]
        for i in range(len(seed)):
            cands = [pid[i]] + cand_generate(predicted_matrix[i], seed[i], reader_challenge.id2uri)
            total_cands.append(cands)

        if reader_challenge.ch_idx == 0:
            break

    with open(conf.result, 'wb') as f:
        pickle.dump(total_cands, f)
#    df = pd.DataFrame(total_cands)
#    df.to_csv('results.csv', index=False, header=False)
