"""
Generate training set, test set and challenge set.
Convert the MPD's data format into our models'.
"""
import sys
import json
import os
import numpy as np
import argparse
from utils.spotify_reader import *


def fullpaths_generator(path):
    filenames = os.listdir(path)
    fullpaths = []
    for filename in filenames:
        fullpath = os.sep.join((path, filename))
        fullpaths.append(fullpath)
    return fullpaths


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--datadir', type=str, default='./data')
    args.add_argument('--mpd_tr', type=str, default='./data-train', help="train data path")
    args.add_argument('--mpd_te', type=str, default='./data-test', help="test data path")
    args.add_argument('--mpd_ch', type=str, default='./challenge', help="challenge data path")
    args.add_argument('--mincount_trk', type=int, default=5, help='minimum count of tracks')
    args.add_argument('--mincount_art', type=int, default=3, help='minimum count of artists')
    
    config = args.parse_args()

    train_fullpaths = fullpaths_generator(config.dpath_tr)
    train_fold = Spotify_train(train_fullpaths, config.mincount_trk, config.mincount_art, True, config.folddir)

    if config.dpath_te != 'NULL':
        test_fullpaths = fullpaths_generator(config.dpath_te)
        for test_seed_num in [0, 1, 5, 10, 25, 100]:
            test_fold = Spotify_test(test_fullpaths, config.folddir+'/train', test_seed_num, config.folddir, False)
            del test_fold

        for test_seed_num in [25, 100]:
            test_fold = Spotify_test(test_fullpaths, config.folddir+'/train', test_seed_num, config.folddir, True)
            del test_fold


    if config.dpath_ch != 'NULL':
        challenge_fullpaths = fullpaths_generator(config.dpath_ch)
        challange_fold = Spotify_challenge(challenge_fullpaths, config.folddir+'/train',
                                           config.folddir, list(range(0,11)), True)
        challange_fold = Spotify_challenge(challenge_fullpaths, config.folddir + '/train',
                                           config.folddir, list(range(25, 101)), True)
        challange_fold = Spotify_challenge(challenge_fullpaths, config.folddir + '/train',
                                           config.folddir, list(range(25, 101)), False)

